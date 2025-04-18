from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
from app import PremiumFiscalAssistant

# Initialisation FastAPI
app = FastAPI(
    title="Assistant Fiscal Sénégalais",
    description="API spécialisée dans la fiscalité sénégalaise - Réponses strictement limitées aux questions fiscales",
    version="1.0.0"
)

# Initialisation de l'assistant fiscal
assistant = PremiumFiscalAssistant()

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fiscal-api")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ... ton code FastAPI reste inchangé au-dessus

@app.get("/fiscalite", summary="Poser une question fiscale", response_description="Réponse de l'assistant fiscal")
async def get_fiscalite(
    question: str = Query(
        ...,
        min_length=3,
        max_length=500,
        description="Posez votre question sur la fiscalité sénégalaise (impôts, taxes, déclarations, etc.)",
        example="Quelles sont les démarches pour obtenir un quitus fiscal ?"
    ),
    strict: Optional[bool] = Query(
        True,
        description="Mode strict (True par défaut) - Ne répond qu'aux questions clairement fiscales"
    )
):
    try:
        # Liste de salutations simples
        salutations = [
            "bonjour", "salut", "hello", "bonsoir", "hi", "coucou",
            "yo", "allo", "bjr"
        ]

        question_lower = question.strip().lower()

        if question_lower in salutations:
            return {
                "reponse": "👋 Bonjour ! Je suis là pour répondre à vos questions fiscales. Que puis-je faire pour vous ?"
            }

        # Validation de la question
        if len(question_lower) < 3:
            raise HTTPException(
                status_code=400,
                detail="La question doit contenir au moins 3 caractères"
            )

        logger.info(f"[QUESTION RECUE] {question}")
        
        # Traitement par l'assistant fiscal
        response = assistant.agent.invoke({"input": question})
        print("🔍 DEBUG - Réponse brute de l'agent:", response)

        response_content = response['output']
        
        # # Vérification supplémentaire en mode strict
        # if strict and not assistant._est_question_fiscale(question):
        #     raise HTTPException(status_code=400, detail="Question non fiscale...")
                
        # logger.info(f"[REPONSE ENVOYEE] {response_content[:200]}...")
        return {"reponse": response_content}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERREUR] {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue lors du traitement de votre question. Veuillez réessayer plus tard."
        )


@app.get("/sante", summary="Vérification rapide de l'état de l'API")
async def check_health():
    try:
        # Vérification Elasticsearch
        if not assistant.es.ping():
            raise HTTPException(
                status_code=503,
                detail="Elasticsearch indisponible"
            )

        # Test minimal du modèle
        test_response = assistant.recherche_fiscale("test santé")
        if not isinstance(test_response, str) or len(test_response) == 0:
            raise HTTPException(
                status_code=503,
                detail="Le modèle fiscal ne répond pas correctement"
            )

        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur système: {str(e)}"
        )
