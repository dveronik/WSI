import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch_directml
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import pandas as pd
from win32gui import GradientFill

##############################################################################

naprava = torch_directml.device()
model = r'.\sloberta.2.0.transformers'
naj_klasifikator = "./best_model_WSI_CA-88.85.ckpt"
max_st_besed = 128
podatki_elexis = "elexis-wsd-sl_corpus.tsv"


##############################################################################

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
            brez_presledka = True if "SpaceAfter=No" in sestavni_deli[2] or "SpaceAfter=No" in sestavni_deli[
                3] or "SpaceAfter=No" in \
                                     sestavni_deli[4] else False
            oznake = sestavni_deli[4]
            trenutna_vrstica.append([idx, beseda, lema, brez_presledka, oznake])

    return elexis_slovar


##############################################################################

tokenizer = CamembertTokenizer.from_pretrained(model, use_fast=False)
dodatna_oznaka = {'additional_special_tokens': ['<target>', '</target>']}
tokenizer.add_special_tokens(dodatna_oznaka)

model_wic = CamembertForSequenceClassification.from_pretrained(model, num_labels=2)
model_wic.resize_token_embeddings(len(tokenizer))

print(f"Nalagam shranjene uteži iz: {naj_klasifikator}")
model_wic.load_state_dict(torch.load(naj_klasifikator, map_location=naprava))
model_wic.to(naprava)
model_wic.eval()

##############################################################################

WSI_mnozica = pd.read_csv("WSI_podatkovna_mnozica.csv", sep=";", encoding="utf-8-sig")
ciljne_besede = set(WSI_mnozica['Beseda'].astype(str).str.lower().str.strip().tolist())
print(f"Moja podatkovna množica vsebuje {len(ciljne_besede)} unikatnih besed.")

celoten_elexis_slovar = preberi_elexis(podatki_elexis)
koncni_elexis_slovar = {}
for lema, pomeni in celoten_elexis_slovar.items():
    if lema.lower().strip() in ciljne_besede:
        koncni_elexis_slovar[lema] = pomeni
print(f"{len(koncni_elexis_slovar)} besed se ujema z besedami v podatkovni množici")

##############################################################################

# Preizkus izpuščanja pomenov besed

odstranjeni_pomeni = []
ohranjeni_pomeni = []

print("\n Začenjam preizkus izpuščanja pomenov")
for lema, pomeni_leme in tqdm(koncni_elexis_slovar.items()):
    if len(pomeni_leme) < 2:
        continue
    vsi_pomeni_idx = list(pomeni_leme.keys())

    for pravi_pomen_id, testne_povedi in pomeni_leme.items():
        pari = []
        id_pomenov = []
        for testna_poved in testne_povedi:
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
            pari = [par for par, primerjalni_pomen_id in zip(pari, id_pomenov) if
                    primerjalni_pomen_id != pravi_pomen_id]

        if not pari:
            continue

        batch_size = 16
        vsi_logiti_povedi = []

        for i in range(0, len(pari), batch_size):
            batch_pari = pari[i: i + batch_size]

            test_encodings = tokenizer(batch_pari, padding='max_length', truncation=True, max_length=max_st_besed,
                                       return_tensors="pt").to(naprava)

            with torch.no_grad():
                izvodi = model_wic(**test_encodings)
                logiti_istega = izvodi.logits[:, 1].cpu().numpy()
                vsi_logiti_povedi.extend(logiti_istega)

        najvisja_napoved = np.max(vsi_logiti_povedi) if vsi_logiti_povedi else 0.0

        if izpust_pomena:
            odstranjeni_pomeni.append(najvisja_napoved)
        else:
            ohranjeni_pomeni.append(najvisja_napoved)


def accuracy(odstranjeni_pomeni, ohranjeni_pomeni, meja):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for logit in odstranjeni_pomeni:
        if logit < meja:
            TP += 1
        else:
            FP += 1

    for logit in ohranjeni_pomeni:
        if logit > meja:
            TN += 1
        else:
            FN += 1
    vsota_klasifikacij = TP + TN + FP + FN
    classification_accuracy = (TP + TN) / vsota_klasifikacij if vsota_klasifikacij > 0 else 0
    return classification_accuracy


mozne_meje = np.unique(odstranjeni_pomeni + ohranjeni_pomeni)
najboljsa_meja = 0
max_accuracy = 0
vse_tocnosti = []
for meja in mozne_meje:
    trenuten_accuracy = accuracy(odstranjeni_pomeni, ohranjeni_pomeni, meja)
    vse_tocnosti.append(trenuten_accuracy)  # Shranimo za graf
    if trenuten_accuracy > max_accuracy:
        max_accuracy = trenuten_accuracy
        najboljsa_meja = meja

##############################################################################


plt.figure(figsize=(10, 6))
plt.plot(mozne_meje, vse_tocnosti, label="Točnost", color="blue", linewidth=2)
plt.axvline(x=najboljsa_meja, color="red", linestyle="--", alpha=0.7,
            label=f"Optimalna meja ({najboljsa_meja:.4f})")
plt.scatter([najboljsa_meja], [max_accuracy], color="red", zorder=5,
            label=f"Max Accuracy ({max_accuracy * 100:.2f}%)")
plt.title("Optimizacija empirične meje glede na točnost", fontsize=14, pad=15)
plt.xlabel("Vrednost empirične meje (Logit threshold)", fontsize=12)
plt.ylabel("Točnost (Accuracy)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11, loc="lower center")
plt.show()

empiricna_meja = najboljsa_meja

TP = 0
TN = 0
FP = 0
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

vsota_evalviranih_primerov = TP + TN + FP + FN
accuracy = (TP + TN) / vsota_evalviranih_primerov if vsota_evalviranih_primerov > 0 else 0

##############################################################################
print("\n" + "=" * 60)
print("Končni rezultati preizkusa izpuščanje pomenov besed:")
print("=" * 60)
print(f"True Positives:  {TP}")
print(f"False Positives:        {FP}")
print(f"True Negatives:  {TN}")
print(f"False Negatives:         {FN}")
print("-" * 60)
print(f"WSI Accuracy: {accuracy * 100:.2f} %")
print("=" * 60)