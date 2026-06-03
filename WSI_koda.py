import torch
import classla
from transformers import AutoModel, AutoTokenizer, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertTokenizer, CamembertForMaskedLM, CamembertModel, CamembertForCausalLM, CamembertConfig
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

podatki = pd.read_csv("WSI_podatkovna_mnozica.csv", sep=";", encoding="utf-8-sig")
lematizator = classla.Pipeline('sl', processors='tokenize,pos,lemma')

# Postopek označevanja ciljne besede v povedih

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

vmesni_slovar = {}
for _, vrstica in podatki.iterrows():
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

sskj_slovar = {}
for beseda in vmesni_slovar:
    pomeni = []
    for pomen in vmesni_slovar[beseda].values():
        pomeni.append(pomen)
    pomeni.sort(key=lambda p: p["pomen_id"])
    sskj_slovar[beseda] = pomeni

##############################################################################

naprava = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = r'D:\DigiLing\MAGISTRSKA\Koda\sloberta.2.0.transformers'
tokenizer = CamembertTokenizer.from_pretrained(model, use_fast=False)

# Postopek tvorjenja učnih parov povedi

def izberi_razlicen_pomen(podatki, beseda, trenutni_i):
    if beseda not in podatki or len(podatki[beseda]) <= 1:
        return None
    vsi_pomeni = podatki[beseda]
    kandidati = []
    for i, pomen in enumerate(vsi_pomeni):
        if i != trenutni_i and len(pomen["povedi"]) > 0:
            kandidati.append(pomen)
    if not kandidati:
        return None
    izbran_pomen = random.choice(kandidati)
    return random.choice(izbran_pomen["povedi"])

def pripravi_ucne_pare(podatki, st_parov_na_pomen=2):
    x_povedi = []  # Seznam parov povedi (npr. "Poved 1 </s> Poved 2")
    y_oznake = []  # Seznam oznak: 1 (isti pomen), 0 (različen pomen)
    for beseda, pomeni in tqdm(podatki.items()):
        for i, pomen in enumerate(pomeni):
            povedi_trenutni_pomen = pomen["povedi"]
            if len(pomeni) < 2 or len(povedi_trenutni_pomen) < 2:
                continue

            ustvarjeni = 0
            poskusi = 0
            while ustvarjeni < st_parov_na_pomen and poskusi < 20:
                poskusi += 1
                izbrani_par = random.sample(povedi_trenutni_pomen, 2)
                poved_1_oznacena = izbrani_par[0]
                poved_2_isti_oznacena = izbrani_par[1]
                poved_2_razlicen_oznacena = izberi_razlicen_pomen(podatki, beseda, i)

                if poved_2_razlicen_oznacena is not None:
                    # Pozitiven par (isti pomen)
                    x_povedi.append(f"{poved_1_oznacena} </s> {poved_2_isti_oznacena}")
                    y_oznake.append(1)
                    # Negativen par (različen pomen)
                    x_povedi.append(f"{poved_1_oznacena} </s> {poved_2_razlicen_oznacena}")
                    y_oznake.append(0)
                    ustvarjeni += 1

    x_povedi, y_oznake = shuffle(x_povedi, y_oznake, random_state=42)
    return x_povedi, y_oznake

##############################################################################

# Učna in testna množica

x_povedi, y_oznake = pripravi_ucne_pare(sskj_slovar)
x_train, x_test, y_train, y_test = train_test_split(x_povedi, y_oznake, train_size=0.8, shuffle=True, random_state=42)
dodatna_oznaka = {'additional_special_tokens': ['<target>', '</target>']}
tokenizer.add_special_tokens(dodatna_oznaka)
max_st_besed = 128

def tokenizacija(sez_povedi, tokenizator, max_dolzina):
    return tokenizator(sez_povedi, padding='max_length', truncation=True,  max_length=max_dolzina,  return_tensors="pt")

ucne_povedi = tokenizacija(x_train, tokenizer, max_st_besed)
testne_povedi = tokenizacija(x_test, tokenizer, max_st_besed)

ucne_oznake = torch.tensor(y_train)
testne_oznake = torch.tensor(y_test)

##############################################################################

# Učenje modela

batch_size = 16

train_dataset = TensorDataset(ucne_povedi['input_ids'], ucne_povedi['attention_mask'], ucne_oznake)
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

model_wsi = CamembertForSequenceClassification.from_pretrained(model, num_labels=2)
model_wsi.resize_token_embeddings(len(tokenizer))
model_wsi.to(naprava)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model_wsi.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model_wsi.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

epochs = 1
running_loss = 0
global_step = 0
model_wsi.train()
for epoch in range(epochs):
    for batch in tqdm(train_dataloader, desc="Učenje modela"):
        global_step += 1
        b_input_ids = batch[0].to(naprava)
        b_input_mask = batch[1].to(naprava)
        b_labels = batch[2].to(naprava).type(torch.LongTensor).to(naprava)

        optimizer.zero_grad()
        outputs = model_wsi(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

        if global_step % 30 == 0:
            print(f"\nKorak {global_step} | Trenutna povprečna izguba (Loss): {running_loss / 30:.4f}", file=sys.stderr)
            running_loss = 0
            torch.save(model_wsi.state_dict(), './trained_model.ckpt')

        del loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    torch.save(model_wsi.state_dict(), './trained_model.ckpt')
    print("\nUčenje modela uspešno zaključeno! Uteži so shranjene v './trained_model.ckpt'.")

##############################################################################

# Evalvacija modela

test_dataset = TensorDataset(testne_povedi['input_ids'], testne_povedi['attention_mask'], testne_oznake)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nZačenjam z evalvacijo na testni množici...")
model_wsi.eval()

correct = 0
incorrect = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testiranje modela"):
        b_input_ids = batch[0].to(naprava)
        b_input_mask = batch[1].to(naprava)
        b_labels = batch[2].to(naprava)

        outputs = model_wsi(input_ids=b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        for pred, true_label in zip(preds, b_labels):
            if pred == true_label:
                correct += 1
            else:
                incorrect += 1

classification_accuracy = correct / (correct + incorrect)

print("\n" + "#" * 50)
print(f" Evalvacija:")
print(f"Število pravilnih napovedi: {correct}")
print(f"Število napačnih napovedi: {incorrect}")
print(f"CA: {classification_accuracy * 100:.2f} %")
print("#" * 50)