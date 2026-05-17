import classla
import pandas as pd

##############################################################################
# 1. NALAGANJE PODATKOV

vhodna_dat = "WSI_podatkovna_mnozica.csv"
izhodna_dat = "Classla_napake.csv"
classla_lematizator = classla.Pipeline("sl", processors="tokenize,pos,lemma")
wsi_podatki = pd.read_csv(vhodna_dat, sep=";", encoding="utf-8-sig")

##############################################################################
# 2. PREVERJANJE POVEDI (Funkcija)

def preveri_poved(poved, ciljna_beseda):
    if pd.isna(poved) or pd.isna(ciljna_beseda):
        return False

    analiza_besedila = classla_lematizator(str(poved))
    osnovne_oblike = []
    for stavek in analiza_besedila.sentences:
        for beseda in stavek.words:
            if beseda.lemma:
                osnovna_oblika = beseda.lemma.lower()
                osnovne_oblike.append(osnovna_oblika)
    if ciljna_beseda.lower() in osnovne_oblike:
        return True
    else:
        return False

##############################################################################
# 3. ZAZNAVANJE NAPAK

seznam_povedi = list(wsi_podatki["Poved"])
seznam_besed = list(wsi_podatki["Beseda"])
st_vrstic = len(wsi_podatki)
vrstice_z_napakami = []
for i in range(len(wsi_podatki)):
    trenutni_stavek = seznam_povedi[i]
    iskana_beseda = seznam_besed[i]
    print(f"Obdelujem vrstico {i + 1}/{st_vrstic}")
    uspesno_zaznano = preveri_poved(trenutni_stavek, iskana_beseda)
    if not uspesno_zaznano:
        trenutna_vrstica = wsi_podatki.iloc[i]
        vrstice_z_napakami.append(trenutna_vrstica)

##############################################################################
# 4. SHRANJEVANJE

dat_napak = pd.DataFrame(vrstice_z_napakami)
dat_napak.to_csv(izhodna_dat, index=False, sep=";", encoding="utf-8-sig")