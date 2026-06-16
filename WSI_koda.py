import json
import os
import time
import torch
import classla
import torch_directml
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import sys
import gc

##############################################################################

print("Nalagam CSV in lematizator")
podatki = pd.read_csv("WSI_podatkovna_mnozica.csv", sep=";", encoding="utf-8-sig")
lematizator = classla.Pipeline('sl', processors='tokenize,pos,lemma')


def oznaci_besedo_in_preveri_poved(poved, ciljna_beseda, pipeline=lematizator):
    analiza_besedila = pipeline(poved)
    oznacena_poved = []
    najdena_ciljna = False
    for stavek in analiza_besedila.sentences:
        for beseda in stavek.words:
            if beseda.lemma.lower() == ciljna_beseda and not najdena_ciljna:
                oznacena_poved.append(f"<target> {beseda.text} </target>")
                najdena_ciljna = True
            else:
                oznacena_poved.append(beseda.text)
    if not najdena_ciljna:
        return None
    return " ".join(oznacena_poved)


def pripravi_slovar():
    sskj_json = "sskj_slovar.json"
    if os.path.exists(sskj_json):
        print("Nalagam slovar iz JSON predpomnilnika")
        with open(sskj_json, "r", encoding="utf-8") as f:
            return json.load(f)

    vmesni_slovar = {}
    len_vsi_podatki = len(podatki)
    print(f"Število vrstic za obdelavo: {len_vsi_podatki}")

    zacetek = time.time()
    for i, (_, vrstica) in enumerate(podatki.iterrows()):
        beseda = str(vrstica['Beseda']).lower().strip()
        pomen_id = vrstica['Pomen_ID']
        pomen = str(vrstica['Razlaga']).strip()
        poved = str(vrstica['Poved']).strip()

        oznacena_poved = oznaci_besedo_in_preveri_poved(poved, beseda)
        if oznacena_poved is None:
            continue

        if beseda not in vmesni_slovar:
            vmesni_slovar[beseda] = {}
        if pomen not in vmesni_slovar[beseda]:
            vmesni_slovar[beseda][pomen] = {"pomen_id": pomen_id, "pomenska_razlaga": pomen, "povedi": []}
        if oznacena_poved not in vmesni_slovar[beseda][pomen]["povedi"]:
            vmesni_slovar[beseda][pomen]["povedi"].append(oznacena_poved)

        if i % 100 == 1:
            konec = time.time()
            delez = i / len_vsi_podatki
            preostalo = ((konec - zacetek) / delez) - (konec - zacetek)
            print(f"Čas: {(konec - zacetek) / 60:.2f} min. Še približno: {preostalo / 60:.2f} min.")

    sskj_slovar = {}
    for beseda, pomeni_dict in vmesni_slovar.items():
        pomeni = list(pomeni_dict.values())
        pomeni.sort(key=lambda p: p["pomen_id"])
        sskj_slovar[beseda] = pomeni

    print("Shranjujem slovar v JSON predpomnilnik")
    with open(sskj_json, "w", encoding="utf-8") as f:
        json.dump(sskj_slovar, f, indent=4, ensure_ascii=False)
    return sskj_slovar


print("Pripravljam slovar")
sskj_slovar = pripravi_slovar()

##############################################################################

print("Povezujem napravo in nalagam tokenizator")
naprava = torch_directml.device()
model_pot = r'.\sloberta.2.0.transformers'
tokenizer = CamembertTokenizer.from_pretrained(model_pot, use_fast=False)
dodatna_oznaka = {'additional_special_tokens': ['<target>', '</target>']}
tokenizer.add_special_tokens(dodatna_oznaka)


def izberi_razlicen_pomen(celoten_slovar, beseda, trenutni_i):
    if beseda not in celoten_slovar or len(celoten_slovar[beseda]) <= 1:
        return None
    vsi_pomeni = celoten_slovar[beseda]
    kandidati = []
    for i, pomen in enumerate(vsi_pomeni):
        povedi_kandidata = pomen["povedi"][:6]
        if i != trenutni_i and len(povedi_kandidata) > 0:
            kandidati.append(povedi_kandidata)
    if not kandidati:
        return None
    izbran_pomen_povedi = random.choice(kandidati)
    return random.choice(izbran_pomen_povedi)

def pripravi_ucne_pare(podmnozica, celoten_slovar, max_primerov_na_pomen=6, max_parov_na_pomen=100):
    x_povedi = []
    y_oznake = []

    for beseda in tqdm(podmnozica, desc="Generiranje parov"):
        pomeni = celoten_slovar[beseda]
        for i, pomen in enumerate(pomeni):
            povedi_trenutni_pomen = pomen["povedi"][:max_primerov_na_pomen]
            if len(pomeni) < 2 or len(povedi_trenutni_pomen) < 2:
                continue

            pozitivni_pari = []
            negativni_pari = []

            for m in range(len(povedi_trenutni_pomen)):
                for n in range(m + 1, len(povedi_trenutni_pomen)):
                    pozitivni_pari.append((povedi_trenutni_pomen[m], povedi_trenutni_pomen[n]))

            for pov_1 in povedi_trenutni_pomen:
                for _ in range(len(povedi_trenutni_pomen)):
                    pov_2_razlicen = izberi_razlicen_pomen(celoten_slovar, beseda, i)
                    if pov_2_razlicen:
                        negativni_pari.append((pov_1, pov_2_razlicen))

            st_parov = min(len(pozitivni_pari), len(negativni_pari), max_parov_na_pomen // 2)
            if st_parov == 0:
                continue

            izbrani_pozitivni = random.sample(pozitivni_pari, st_parov)
            izbrani_negativni = random.sample(negativni_pari, st_parov)

            for p_1, p_2 in izbrani_pozitivni:
                x_povedi.append(f"{p_1} </s> {p_2}")
                y_oznake.append(1)
            for n_1, n_2 in izbrani_negativni:
                x_povedi.append(f"{n_1} </s> {n_2}")
                y_oznake.append(0)

    if len(x_povedi) == 0:
        return [], []
    x_povedi, y_oznake = shuffle(x_povedi, y_oznake, random_state=42)
    return x_povedi, y_oznake


##############################################################################

vse_besede = list(sskj_slovar.keys())
x_vsi_pari, y_vse_oznake = pripravi_ucne_pare(vse_besede, sskj_slovar, max_primerov_na_pomen=6,
                                              max_parov_na_pomen=100)

print(f"Skupno število zgeneriranih parov: {len(x_vsi_pari)}")

x_train, x_test, y_train, y_test = train_test_split(
    x_vsi_pari,
    y_vse_oznake,
    train_size=0.8,
    shuffle=True,
    random_state=42
)

print(f"Pari razdeljeni: {len(x_train)} za učenje, {len(x_test)} za testiranje")

max_st_besed = 128

def tokenizacija(sez_povedi, tokenizator, max_dolzina):
    return tokenizator(sez_povedi, padding='max_length', truncation=True, max_length=max_dolzina, return_tensors="pt")

ucne_povedi = tokenizacija(x_train, tokenizer, max_st_besed)
testne_povedi = tokenizacija(x_test, tokenizer, max_st_besed)

ucne_oznake = torch.tensor(y_train)
testne_oznake = torch.tensor(y_test)

##############################################################################

batch_size = 16

train_dataset = TensorDataset(ucne_povedi['input_ids'], ucne_povedi['attention_mask'], ucne_oznake)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(testne_povedi['input_ids'], testne_povedi['attention_mask'], testne_oznake)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Nalagam model SloBERTa")
model_wsi = CamembertForSequenceClassification.from_pretrained(model_pot, num_labels=2)
model_wsi.resize_token_embeddings(len(tokenizer))
model_wsi.to(naprava)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model_wsi.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model_wsi.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

##############################################################################

print("Začenjam učenje modela SloBERTa")
epochs = 20
best_accuracy = 0.0
global_step = 0

for epoch in range(epochs):
    model_wsi.train()
    running_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoha {epoch + 1}/{epochs} [Učenje]"):
        global_step += 1
        b_input_ids = batch[0].to(naprava)
        b_input_mask = batch[1].to(naprava)
        b_labels = batch[2].to(naprava).long()

        optimizer.zero_grad()
        outputs = model_wsi(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if global_step % 30 == 0:
            torch.save(model_wsi.state_dict(), './trained_model.ckpt')

        del loss
        gc.collect()

    model_wsi.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids = batch[0].to(naprava)
            b_input_mask = batch[1].to(naprava)
            b_labels = batch[2].to(naprava)

            outputs = model_wsi(input_ids=b_input_ids, attention_mask=b_input_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            # Popravek za stabilnost DirectML primerjave
            correct += (preds.cpu() == b_labels.cpu()).sum().item()
            total += b_labels.size(0)

    epoch_accuracy = correct / total if total > 0 else 0
    avg_train_loss = running_loss / len(train_dataloader)

    print(f"\n--- Epoha {epoch + 1} zaključena ---")
    print(f"Učna izguba: {avg_train_loss:.4f}")
    print(f"Točnost na testnih parih: {epoch_accuracy * 100:.2f} %")

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        ime_najboljsega = f"./best_model_WSI_CA-{best_accuracy * 100:.2f}.ckpt"

        torch.save(model_wsi.state_dict(), './trained_model.ckpt')
        torch.save(model_wsi.state_dict(), ime_najboljsega)
        print(f"Shranjen nov najboljši model: {ime_najboljsega}")
    print("-" * 40)

torch.save(model_wsi.state_dict(), './trained_model.ckpt')
print(f"\nUčenje končano. Najvišji CA: {best_accuracy * 100:.2f} %")