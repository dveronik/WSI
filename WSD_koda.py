import torch
import classla
from transformers import AutoModel, AutoTokenizer, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertTokenizer, CamembertForMaskedLM, CamembertModel, CamembertForCausalLM, CamembertConfig
import pandas as pd
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

##############################################################################

# 1. PODATKOVNA MNOŽICA
podatki = pd.read_csv("testni_dataset.csv", sep=";", encoding="utf-8-sig")

vmesni_slovar = {}
for _, vrstica in podatki.iterrows():
    beseda = str(vrstica['Beseda']).lower().strip()
    pomen_id = vrstica['Pomen_ID']
    pomen = str(vrstica['Razlaga']).strip()
    poved = str(vrstica['Poved']).strip()
    if beseda not in vmesni_slovar:
        vmesni_slovar[beseda] = {}
    if pomen not in vmesni_slovar[beseda]:
        vmesni_slovar[beseda][pomen] = {"pomen_id": pomen_id, "pomenska_razlaga": pomen, "povedi": []}
    vmesni_slovar[beseda][pomen]["povedi"].append(poved)

sskj_slovar = {}
for beseda in vmesni_slovar:
    pomeni = []
    for pomen in vmesni_slovar[beseda].values():
        pomeni.append(pomen)
    pomeni.sort(key=lambda p: p["pomen_id"])
    sskj_slovar[beseda] = pomeni

##############################################################################

max_st_besed = 300
naprava = torch.device("cpu")
model = r'D:\DigiLing\MAGISTRSKA\Koda\sloberta.2.0.transformers'
# tokenizer = CamembertTokenizer.from_pretrained(model, use_fast=False)
lematizator = classla.Pipeline('sl', processors='tokenize,pos,lemma')

# 2. PRIPRAVA POVEDI

def oznaci_ciljno_besedo(poved, ciljna_beseda, pipeline=lematizator):
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


def pripravi_podatke(podatki, st_parov_na_pomen=2):
    x = []  # Seznam parov povedi (npr. "Poved 1 </s> Poved 2")
    y = []  # Seznam oznak: 1 (isti pomen), 0 (različen pomen)
    for beseda, pomeni in tqdm(podatki.items()):
        for i, pomen in enumerate(pomeni):
            povedi_trenutni_pomen = vnos["povedi"]
            if len(pomeni) < 2 or len(povedi_trenutni_pomen) < 2:
                continue

            ustvarjeni = 0
            poskusi = 0
            while ustvarjeni < st_parov_na_pomen and poskusi < 20:
                poskusi += 1
                izbrani_par = random.sample(povedi_trenutni_pomen, 2)
                poved_1 = izbrani_par[0]
                poved_2_isti = izbrani_par[1]
                poved_2_razlicen = izberi_razlicen_pomen(podatki, beseda, i)
                if poved_2_razlicen:
                    poved_1_oznacena = oznaci_ciljno_besedo(poved_1, beseda)
                    poved_2_isti_oznacena = oznaci_ciljno_besedo(poved_2_isti, beseda)
                    poved_2_razlicen_oznacena = oznaci_ciljno_besedo(poved_2_razlicen, beseda)

                    if poved_1_oznacena and poved_2_isti_oznacena and poved_2_razlicen_oznacena:
                        # Pozitiven par (isti pomen)
                        x.append(f"{poved_1_oznacena} </s> {poved_2_isti_oznacena}")
                        y.append(1)
                        # Negativen par (različen pomen)
                        x.append(f"{poved_1_oznacena} </s> {poved_2_razlicen_oznacena}")
                        y.append(0)
                        ustvarjeni += 1

    x, y = shuffle(x, y, random_state=42)
    return x, y

































# --- 3. GLAVNA ZANKA ZA GENERIRANJE WiC PAROV ---

X = []  # Seznam za pare: "Poved1 </s> Poved2"
Y = []  # Seznam za oznake: 1 (isti pomen), 0 (različen pomen)

# Koliko parov želimo ustvariti za vsak pomen
# Zvišaj to številko, če želiš večji dataset za trening
stevilo_parov_na_pomen = 3

print("Začenjam generiranje WiC parov (to lahko traja zaradi Classle)...")

# sskj_slovar je tvoj strukturiran slovar iz 1. dela
for beseda, seznam_pomenov in tqdm(sskj_slovar.items()):
    for i, vnos in enumerate(seznam_pomenov):
        povedi_tega_pomena = vnos["povedi"]

        # --- A) ISTI POMEN (Label 1) ---
        if len(povedi_tega_pomena) >= 2:
            st_poskusov = 0
            uspesni_pari = 0
            while uspesni_pari < stevilo_parov_na_pomen and st_poskusov < 10:
                st_poskusov += 1
                par_raw = random.sample(povedi_tega_pomena, 2)

                p1 = vstavi_target_zeton(par_raw[0], beseda, lematizator)
                p2 = vstavi_target_zeton(par_raw[1], beseda, lematizator)

                if p1 and p2:
                    X.append(f"{p1} </s> {p2}")
                    Y.append(1)
                    uspesni_pari += 1

        # --- B) RAZLIČEN POMEN (Label 0) ---
        if len(seznam_pomenov) >= 2:
            st_poskusov = 0
            uspesni_pari = 0
            while uspesni_pari < stevilo_parov_na_pomen and st_poskusov < 10:
                st_poskusov += 1
                poved_a_raw = random.choice(povedi_tega_pomena)
                poved_b_raw = izberi_poved_drugega_pomena(sskj_slovar, beseda, i)

                if poved_b_raw:
                    p1 = vstavi_target_zeton(poved_a_raw, beseda, lematizator)
                    p2 = vstavi_target_zeton(poved_b_raw, beseda, lematizator)

                    if p1 and p2:
                        X.append(f"{p1} </s> {p2}")
                        Y.append(0)
                        uspesni_pari += 1

# --- 4. RAZDELITEV NA TRENING IN VALIDACIJO (80/20) ---

print(f"\nSkupaj ustvarjenih parov: {len(X)}")

X_train, X_eval, y_train, y_eval = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Velikost učne množice (Train): {len(X_train)}")
print(f"Velikost validacijske množice (Eval): {len(X_eval)}")

# --- TESTNI IZPIS PRVEGA PARA ---
if len(X_train) > 0:
    print("\nPrimer prvega para v X_train:")
    print(X_train[0])
    print(f"Oznaka (Y): {y_train[0]}")


# ###############################################################
#
# X = []  # Seznam za pare povedi
# Y = []  # Seznam za oznake (1 ali 0)
#
# # Gremo čez vsako besedo v našem novem slovarju
# for beseda in sskj_dict:
#     vsi_pomeni_ids = list(sskj_dict[beseda].keys())
#     lema_besede = lematizator.lemmatize(beseda)
#
#     for p_id in vsi_pomeni_ids:
#         povedi_tega_pomena = sskj_dict[beseda][p_id]["zgledi"]
#
#         # Ustvarimo pare z ISTIM pomenom (Label 1)
#         if len(povedi_tega_pomena) >= 2:
#             poved1 = povedi_tega_pomena[0]
#             poved2 = povedi_tega_pomena[1]
#             # Združimo ju s posebnim žetonom </s>
#             par_povedi = poved1 + " </s> " + poved2
#             X.append(par_povedi)
#             Y.append(1)
#
#         # Ustvarimo pare z RAZLIČNIM pomenom (Label 0)
#         if len(vsi_pomeni_ids) >= 2:
#             # Izberemo drug pomen iste besede
#             drug_p_id = vsi_pomeni_ids[0]
#             if drug_p_id == p_id:
#                 drug_p_id = vsi_pomeni_ids[1]
#
#             poved_drugega_pomena = sskj_dict[beseda][drug_p_id]["zgledi"][0]
#             par_povedi_razlicno = povedi_tega_pomena[0] + " </s> " + poved_drugega_pomena
#             X.append(par_povedi_razlicno)
#             Y.append(0)
#
#
# ##############################################
#
# # Naložimo model
# model = CamembertForSequenceClassification.from_pretrained(pot_do_modela, num_labels=2)
# model.to(naprava)
# model.train()
#
# optimizator = AdamW(model.parameters(), lr=1e-5)
# velikost_paketa = 8  # Batch size
#
# # Glavna zanka za trening
# for i in range(0, len(X), velikost_paketa):
#     # Pripravimo paket podatkov
#     paket_X = X[i: i + velikost_paketa]
#     paket_Y = Y[i: i + velikost_paketa]
#
#     if len(paket_X) == 0:
#         continue
#
#     # Tokenizacija (pretvori besedilo v številke za model)[cite: 2]
#     vhodni_podatki = tokenizer(
#         paket_X,
#         padding='max_length',
#         truncation=True,
#         max_length=MAX_SEQ_LEN,
#         return_tensors="pt"
#     ).to(naprava)
#
#     tarče = torch.tensor(paket_Y).to(naprava)
#
#     # Izračun modela
#     optimizator.zero_grad()
#     izhod = model(**vhodni_podatki, labels=tarče)
#     izguba = izhod.loss
#
#     # Posodobitev modela
#     izguba.backward()
#     optimizator.step()
#
#     # Čiščenje pomnilnika, da računalnik ne zamrzne
#     del izhod, izguba
#     gc.collect()
#     torch.cuda.empty_cache()
#
#     if i % 160 == 0:
#         print(f"Obdelano {i} od {len(X)} parov...")
#
# ##########################################
#
# model.eval()
# print("Trening končan. Shranjujem model...")
#
# # Shranimo rezultat tvojega dela
# torch.save(model.state_dict(), './moj_wsd_model.ckpt')