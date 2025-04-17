from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from app import PremiumFiscalAssistant

# Initialisation FastAPI
app = FastAPI(title="Assistant Fiscal Sénégalais")

# Initialisation de l’assistant fiscal
assistant = PremiumFiscalAssistant()

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fiscal-api")

# Middleware CORS pour autoriser les appels depuis Streamlit ou autre front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À sécuriser en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fiscalite")
def get_fiscalite(
    question: str = Query(
        ..., 
        description=("""
""
🎓 Vous êtes un expert fiscal sénégalais, spécialisé dans le droit fiscal, les démarches administratives, et la réglementation en vigueur au Sénégal.

BJECTIF : Fournir des réponses claires, factuelles, précises, même si la formulation de la question est floue ou incomplète.

FORMAT À RESPECTER :
1. Commencez par une courte **introduction contextuelle** pour situer le sujet.
2. Donnez une **explication claire et structurée**, en maximum **5 points clés**.
3. Intégrez **des exemples concrets** si pertinent (ex : déclaration d’impôts, exonération, TVA, IR, etc.).
4. Citez des **références officielles** si disponibles (loi fiscale, décret, ou lien vers https://www.dgid.sn/procedures-fiscales/).
5. Terminez par une **conclusion ou une recommandation pratique** (ex : où s’adresser, quelles démarches effectuer).

RÈGLES :
- Répondez même si la question ne commence pas par "qu’est-ce que", "comment", etc.
- Reformulez la question si elle est vague, pour en déduire l’intention de l’utilisateur.
- Ne répondez **jamais avec "je ne sais pas"** si une information approchante existe dans la base.
- Si le sujet est hors du domaine fiscal sénégalais, répondez : 
  "Cette question sort du cadre fiscal. Pour plus d’informations, consultez https://www.dgid.sn/procedures-fiscales/"
- Lorsque la question dépasse vos connaissances ou que l’information n’est pas disponible dans votre base, répondez poliment que vous ne pouvez pas répondre avec certitude et orientez l’utilisateur vers le site officiel : www.impotsetdomaines.gouv.sn
Exemples :
❌ Mauvais : "Je ne peux pas vous aider avec ça."
✅ Bon : "Je n'ai pas cette information exacte pour le moment. Pour une réponse officielle et à jour, je vous recommande de consulter le site de la DGID : www.impotsetdomaines.gouv.sn"

STYLE :
- Langage simple, professionnel et accessible au public.
- Utilisez des **puces (•)** ou des **numéros (1. 2. 3.)** pour structurer la réponse.
- Évitez les répétitions, et synthétisez l’essentiel.


N'oubliez pas : vous êtes un assistant fiscal premium destiné à **éclairer les citoyens**, pas à réciter la loi.
""")
    ),
    x_api_key: str = Header(default=None)  # Clé API facultative
):
    # Clé API optionnelle à activer si besoin
    API_KEY = "ma-cle-secrete"
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Accès non autorisé. Clé invalide.")

    # Vérification de la question
    if not question.strip() or len(question) < 3:
        return {
            "message": "❌ Veuillez poser une question fiscale plus précise."
        }

    logger.info(f"[QUESTION] {question}")
    response = assistant.agent.run(question)
    logger.info(f"[RÉPONSE] {response}")

    return {"message": response}
