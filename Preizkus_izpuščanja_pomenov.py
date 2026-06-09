import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import pandas as pd

##############################################################################

naprava = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "" #pot do natreniranega modela
max_st_besed = 128
podatki_elexis = "elexis-wsd-sl_corpus.tsv"

##############################################################################

# Elexis-WSD podatkovna množica

def elexis_povedi(sez_vrstic):
    vrnjene_povedi = []
    vrnjene_oznake = []
    vrnjene_leme = []
    trenutna_poved = ""
    vsebuje_oznako = True
    trenutni_pomen = None
    trenutna_lema = None

    while vsebuje_oznako:
        vsebuje_oznako = False
        for sestavni_del in sez_vrstic:
            idx = sestavni_del[0]
            beseda = sestavni_del[1]
            lema = sestavni_del[2]
            brez_presledka = sestavni_del[3]
            oznake = sestavni_del[4]
            if len(oznake.split("|")) >= 4 and vsebuje_oznako == False:
                vsebuje_oznako = True
                trenutni_pomen = oznake.split("|")[3]
                trenutna_lema = lema
                trenutna_poved += "<target> "
                sestavni_del[4] = "".join(oznake.split("|")[:2])
            trenutna_poved += beseda
            if not brez_presledka:
                trenutna_poved += " "

        if vsebuje_oznako:
            popravljena_poved = trenutna_poved[:-1]
            besede = popravljena_poved.split()
            if "<target>" in besede:
                idx_oznake = besede.index("<target>")
                if idx_oznake + 1 < len(besede):
                    besede.insert(idx_oznake + 2, "</target>")

            vrnjene_povedi.append(" ".join(besede))
            vrnjene_oznake.append(trenutni_pomen)
            vrnjene_leme.append(trenutna_lema)
        trenutna_poved = ""

    return vrnjene_povedi, vrnjene_oznake, vrnjene_leme

def preberi_elexis(ime_datoteke):
    elexis_slovar = {}
    with open(ime_datoteke, 'r', encoding='utf-8') as datoteka:
        trenutna_vrstica = []
        for vrstica in datoteka:
            if vrstica.startswith("#"):
                continue
            if len(vrstica.strip()) == 0:
                if trenutna_vrstica:
                    trenutne_povedi, trenutne_oznake, trenutne_leme = elexis_povedi(trenutna_vrstica)
                    for poved, pomen_id, lema in zip(trenutne_povedi, trenutne_oznake, trenutne_leme):
                        if lema not in elexis_slovar:
                            elexis_slovar[lema] = {}
                        if pomen_id not in elexis_slovar[lema]:
                            elexis_slovar[lema][pomen_id] = []
                        if poved not in elexis_slovar[lema][pomen_id]:
                            elexis_slovar[lema][pomen_id].append(poved)
                    trenutna_vrstica = []
                continue

            sestavni_deli = vrstica.strip().split('\t')
            if len(sestavni_deli) < 5:
                continue
            idx = sestavni_deli[0]
            beseda = sestavni_deli[1]
            lema = sestavni_deli[2]
            brez_presledka = True if "SpaceAfter=No" in sestavni_deli[2] or "SpaceAfter=No" in sestavni_deli[3] or "SpaceAfter=No" in \
                                        sestavni_deli[4] else False
            oznake = sestavni_deli[4]
            trenutna_vrstica.append([idx, beseda, lema, brez_presledka, oznake])

    return elexis_slovar

##############################################################################

tokenizator = CamembertTokenizer.from_pretrained(model, use_fast=False)
dodatna_oznaka = {'additional_special_tokens': ['<target>', '</target>']}
tokenizator.add_special_tokens(dodatna_oznaka)

model_wsi = CamembertForSequenceClassification.from_pretrained(model, num_labels=2)
model_wsi.resize_token_embeddings(len(tokenizator))
model_wsi.load_state_dict(torch.load('./trained_model.ckpt', map_location=naprava))
model_wsi.to(naprava)
model_wsi.eval()

##############################################################################

# Izluščim ciljne besede iz podatkovne množice WSI (gpt-oss-120b povedi)

WSI_mnozica = pd.read_csv("WSI_podatkovna_mnozica.csv", sep=";", encoding="utf-8-sig")
ciljne_besede = set(WSI_mnozica['Beseda'].astype(str).str.lower().str.strip().tolist())
print(f" WSI podatkovna množica vsebuje {len(ciljne_besede)} unikatnih besed.")

celoten_elexis_slovar = preberi_elexis("elexis-wsd-sl_corpus.tsv")
koncni_elexis_slovar = {}
for lema, pomeni in celoten_elexis_slovar.items():
    if lema.lower().strip() in ciljne_besede:
        koncni_elexis_slovar[lema] = pomeni
print(f"{len(koncni_elexis_slovar)} besed se ujema z besedami v WSI podatkovni množici")
if len(koncni_elexis_slovar) == 0:
    print("Napaka")
    exit()

##############################################################################

# Preizkus izpuščanja pomenov besed

odstranjeni_pomeni = []
ohranjeni_pomeni = []
for lema, pomeni_leme in tqdm(koncni_elexis_slovar.items()):
    if len(pomeni_leme) < 2:
        continue
    vsi_pomeni_idx = list(pomeni_leme.keys())

    for pravi_pomen_id, testne_povedi in pomeni_leme.items():
        pari = []
        id_pomenov = []
        for testna_poved in testne_povedi:
            # Sestavimo primerjalne WiC pare z vsemi pomeni te besede znotraj Elexisa
            for primerjalni_pomen_id, primerjalne_povedi in pomeni_leme.items():
                for primerjalna_poved in primerjalne_povedi:
                    pari.append(f"{testna_poved} </s> {primerjalna_poved}")
                    id_pomenov.append(primerjalni_pomen_id)
            if len(pari) > 60:
                pari = pari[:60]
                id_pomenov = id_pomenov[:60]

        izpust_pomena = False
        if random.random() < 0.5:
            izpust_pomena = True
            pari = [par for par, primerjalni_pomen_id in zip(pari, id_pomenov) if primerjalni_pomen_id != pravi_pomen_id]

        if not pari:
            continue

        # Tokenizacija in prenos vhodnih podatkov na grafično kartico/procesor
        test_encodings = tokenizator(pari, padding='max_length', truncation=True, max_length=max_st_besed,
                                     return_tensors="pt").to(naprava)

        with torch.no_grad():
            izvodi = model_wsi(**test_encodings)
            verjetnosti_istega = izvodi.logits[:, 1].cpu().numpy()
            najvisja_napoved = np.max(verjetnosti_istega)

        if izpust_pomena:
            odstranjeni_pomeni.append(najvisja_napoved)
        else:
            ohranjeni_pomeni.append(najvisja_napoved)


empiricna_meja = 2.976  # Meja gotovosti prevzeta iz profesorjevega poskusa
TP = 0
TN = 0
FP= 0
FN = 0

for logit in odstranjeni_pomeni:
    if logit < empiricna_meja:
        TP += 1
    else:
        FP += 1

for logit in ohranjeni_pomeni:
    if logit > empiricna_meja:
        TN += 1
    else:
        FN += 1

stat_evalviranih_primerov = TP + TN + FP + FN
accuracy = (TP + TN) / stat_evalviranih_primerov if stat_evalviranih_primerov > 0 else 0

##############################################################################
print("\n" + "="*60)
print("Končni rezultati preizkusa izpuščanja pomenov besed:")
print("="*60)
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"True negatives: {TN}")
print(f"False negatives: {FN}")
print("-" * 60)
print(f"WSI Accuracy: {accuracy * 100:.2f} %")
print("="*60)