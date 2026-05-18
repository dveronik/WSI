import json
import pandas as pd
import requests

##############################################################################
# 1. NALAGANJE PODATKOV

vhodna_dat = "CLASSLA_KONCNA.csv"
izhodna_dat = "BESANA_REZULTAT_KONCEN.csv"
amebis_url = "http://localhost:225/api/v2/check"
kategorije_napak = "BESANA_CAT_1,BESANA_CAT_2,BESANA_CAT_3,BESANA_CAT_4,BESANA_CAT_5,BESANA_CAT_6"
wsi_podatki = pd.read_csv(vhodna_dat, sep=";", encoding="utf-8-sig")

##############################################################################
# 2. FUNKCIJA ZA POPRAVLJANJE POVEDI

def popravi_poved(poved):
    if pd.isna(poved) or poved.strip() == "":
        return poved, "Brez napak"

    nastavitve = {"format": "plain","data": json.dumps({"annotation": [{"text": poved}]}),
                  "language": "sl", "enabledCategories": kategorije_napak, "enabledOnly": "false",}

    try:
        odgovor = requests.post(amebis_url, data=nastavitve, timeout=15)

        if odgovor.status_code == 200:
            rezultat_analize = odgovor.json()
            vse_zaznane_napake = rezultat_analize["matches"]

            # 1. KATEGORIJA: Amebis ni zaznal napak
            if len(vse_zaznane_napake) == 0:
                return poved, "Brez napak"

            # Preverimo, ali ima vsaj ena napaka predlog za popravek
            ima_predloge_za_popravek = False
            for napaka in vse_zaznane_napake:
                if "replacements" in napaka:
                    if len(napaka["replacements"]) > 0:
                        ima_predloge_za_popravek = True

            # 2. KATEGORIJA: Amebis je zaznal napako, vendar je ni samodejno popravil
            if not ima_predloge_za_popravek:
                return poved, "Dodatno"

            # 3. KATEGORIJA: Amebis je popravil napake
            vse_zaznane_napake.sort(key=lambda x: x["offset"], reverse=True)

            urejena_poved = poved
            spremenjeno = False
            for napaka in vse_zaznane_napake:
                if "replacements" in napaka:
                    if len(napaka["replacements"]) > 0:
                        vsi_predlogi = napaka["replacements"]
                        izbran_predlog = vsi_predlogi[0]["value"]
                        zacetek_napake = napaka["offset"]
                        dolzina_napake = napaka["length"]
                        urejena_poved = (urejena_poved[:zacetek_napake] + izbran_predlog
                                         + urejena_poved[zacetek_napake + dolzina_napake:])
                        spremenjeno = True

            if spremenjeno:
                return urejena_poved, "Popravljeno"
            else:
                return poved, "Brez napak"
        else:
            print(f"Napaka strežnika: {odgovor.status_code}")
            return poved, "Težava s strežnikom"
    except Exception as e:
        print(f"Težave s povezavo: {e}")
        return poved, "Težava s povezavo"

##############################################################################
# 3. GLAVNA ZANKA ZA PROCESIRANJE CELOTNE WSI PODATKOVNE MNOŽICE

seznam_povedi = list(wsi_podatki["Poved"])
skupno_vrstic = len(wsi_podatki)
nove_vrstice = []
for i in range(skupno_vrstic):
    trenutna_poved = seznam_povedi[i]
    print(f"{i + 1}/{skupno_vrstic}")

    nova_poved, koncni_status = popravi_poved(trenutna_poved)
    trenutna_vrstica = wsi_podatki.iloc[i].copy()
    trenutna_vrstica["Poved"] = nova_poved
    trenutna_vrstica["Kategorija Besana"] = koncni_status
    nove_vrstice.append(trenutna_vrstica)

##############################################################################
# 4. SHRANJEVANJE V NOVO CSV DATOTEKO
koncna_dat = pd.DataFrame(nove_vrstice)
koncna_dat.to_csv(izhodna_dat, index=False, sep=";", encoding="utf-8-sig")
print(f"Končano! Rezultati so shranjeni v: {izhodna_dat}")