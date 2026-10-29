import re
from pathlib import Path
from wordfreq import zipf_frequency

TOK_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent.parent.parent / "resources"

_EMBEDDED_STOPWORDS = """
nahi,nahin,nhi,nai,nahee
hai,he,hy
hain,hn,hen
kya,kia,kyaa
ka,kaa
ki,kee
ke
mein,me,main,mai
se
ko,koo
aur,aor,awr,aaur
bhi,bi,bhee
kyun,kyon,kyu,kyoun,kyoon
kaise,kese,kaisa,kaisi
kitna,kitni,kitne
ho
hota,hote,hoti
karta,karte,karti
ye,yeh
wo,woh,vo,voh
jo,zo
ya
lekin,lekn
isliye,islye
kyunki,kyunke,kynki
liye,lye
par,pr
pe
tha
thi,thee
the
raha,rha
rahi,rhi
rahe,rhe
hoon,hun
kab,kabhi
kahan,kaha
kidhar
kaun,koun
kisi
kuch,kuchh
koi
sab,sabhi
ab,abhi
jab,tab
tabhi
agar
yadi
magar
parantu
balki
taki,taaki
phir,fir
aise,aisa,aisi
waise,waisa,waisi
jaise,jaisa,jaisi
itna,itni,itne
wahan,waha
yahan,yaha
hum,ham
tum,aap
mera,meri
tera,teri,tere
iska,iski,iske
uska,uski,uske
inka,inki,inke
unka,unki,unke
apna,apne,apni
saath,sath
baad
pehle
andar
bahar
aage
peeche,piche
paas
door
bina
tak
gaya,gaye,gayi
hua,hue,hui
sakta,sakte,sakti
chahiye
chuka,chuke,chuki
padta,padte,padti
deta,dete,deti
kare,karen,karega,karenge
hoga,hogi,hoge,honge
kar,karna,karne
dena,dene
lena,lene
ja,jaa,jana,jane,jata,jate,jati
aana,aane,aata,aate,aati
bol,bolna,bolne
pa,pana,pane,paate,paati
bilkul
zyada,jada
kam
bahut,bohot
mat
sahi
toh,to
na
bas
sirf
shuru
khatam
lagta,lagte,lagti
pata
baat
naam
tarah
waqt
din
saal
log,logon
cheez,cheezein
bana,banane,banae
rakh,rakhna,rakhne
dekh,dekhna,dekhne
samajh,samajhna
mil,milna,milne,milta,milte,milti
chhod,chhodna
aao,jao,karo,dekho,bolo
jabki,jabbki
usme,usmein
isme,ismein
usse,ussey
isse,issey
usko
isko
unka,unke,unki
inka,inke,inki
kiya
kiye,kye
diya,diye,dya
liya,liye,lye
gaya,gayi,gaye,gya,gyi,gye
jaata,jaati,jaate,jata,jati,jate
wala,wale,wali
jahan,jaha
unhe,unhein
inhe,inhein
usne
isne
"""

_EMBEDDED_AMBIGUOUS = "main, to, me, hi, is, us, par, so, he, in, an, the"

def tokenize(text: str) -> list[str]:
    return TOK_RE.findall(text.lower())

def load_stopword_groups(path=None) -> dict[str, str]:
    p = Path(path) if path else _DEFAULT_RESOURCES_DIR / "hi_stopwords.txt"
    if not p.exists():
        p = Path("resources/hi_stopwords.txt")
    mapping = {}
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else _EMBEDDED_STOPWORDS.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [w.strip().lower() for w in line.split(",") if w.strip()]
        if parts:
            canonical = parts[0]
            for variant in parts:
                mapping[variant] = canonical
    return mapping

def load_ambiguous(path=None) -> set[str]:
    p = Path(path) if path else _DEFAULT_RESOURCES_DIR / "ambiguous.txt"
    if not p.exists():
        p = Path("resources/ambiguous.txt")
    ambiguous = set()
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else [_EMBEDDED_AMBIGUOUS]
    for line in lines:
        for w in line.split(","):
            w = w.strip().lower()
            if w:
                ambiguous.add(w)
    return ambiguous

class Tagger:
    def __init__(self, lexicon=None, stopwords_path=None, ambiguous_path=None):
        self.lexicon = lexicon
        self.stopword_map = load_stopword_groups(stopwords_path)
        self.stopwords = set(self.stopword_map.keys())
        self.ambiguous = load_ambiguous(ambiguous_path)

    def tag(self, tokens: list[str]) -> list[str]:
        tags = [None] * len(tokens)
        # Pass 1: Tag definite tokens
        is_rule2 = [False] * len(tokens)
        for i, tok in enumerate(tokens):
            t = tok.lower()
            if not t or t.isdigit():
                tags[i] = "OTHER"
            elif t in self.stopwords and t not in self.ambiguous:
                tags[i] = "HI"
                is_rule2[i] = True
            elif t in self.ambiguous:
                tags[i] = None
            elif zipf_frequency(t, "en") >= 3.0:
                tags[i] = "EN"
            elif self.lexicon is not None and self.lexicon.has(t):
                tags[i] = "HI"
            else:
                tags[i] = "EN"

        # Pass 2: Resolve ambiguous tokens based on rule-2 context only
        for i, tok in enumerate(tokens):
            if tags[i] is None:
                left_hi = i > 0 and is_rule2[i - 1]
                right_hi = i + 1 < len(tokens) and is_rule2[i + 1]
                if left_hi or right_hi:
                    tags[i] = "HI"
                else:
                    tags[i] = "EN"

        return tags
