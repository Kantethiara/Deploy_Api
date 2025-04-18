import os
import warnings
import urllib3
from typing import List, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# Désactivation des avertissements
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
load_dotenv()


class PremiumFiscalAssistant:
    def __init__(self):
        self.mots_cles_fiscaux = {
            "impôt", "impot", "taxe", "tva", "TVA", "CFPNB","etax", "cfpnb", "PV", "pv", "PME", "quitus", 
            "PCF", "fiscalité", "déclaration", "CGU", "Patente", "récapitulatifs", "exonération", 
            "remboursement", "trop perçu", "délai", "quitus fiscal", "délai de paiement", 
            "quittance", "récépissé", "revenus", "formalisation", "contribution", "taxation", "cadastre",
            "redevance", "contribution foncière", "taxe sur les véhicules", "taxe sur les biens",
            "taxe sur les opérations", "taxe sur les produits", "taxe sur les services",
            "droit d'enregistrement", "droits d'enregistrement", "taxes d'enregistrement", 
            "entreprise", "changement de statuts", "taxes sur les salaires", "taxe sur les salaires", 
            "taxe foncière", "taxe professionnelle", "NINEA", "direct", "indirect", "réouverture",
            "taxe sur la valeur ajoutée", "passeport", "taxe sur les boissons", "réductions", 
            "immatriculation", "propriétaire", "compte", "duplicata", "IR", "IS", "patente", 
            "douane", "régime fiscal", "code général des impôts", "procédure", 
            "acte administratif", "exonérations", "obligation fiscale", "pénalité", "penalite", 
            "amende", "contrôle fiscal", "démarrage des activités", "homologation", "acte", 
            "titre", "SIGTAS", "imposition", "bail", "foncier bâti", "foncier non bâti", "TEOM", 
            "vérification", "versement", "trésor", "TVA déductible", "TVA collectée", 
            "TVA non récupérable", "non-assujetti", "assujetti", "centre des impôts", 
            "régularisation", "déductibilité", "déclaration mensuelle", "déclaration annuelle", 
            "numéro fiscal", "avis d'imposition", "bordereau de paiement", "numéro IFU", "COFI",
            "fiscale", "fiscaux", "fiscal", "DGID", "impotsetdomaines", "dgi", "direction générale des impôts"
        }
        
        self.es = self._init_elasticsearch()
        self.embedder = self._init_embedder()
        self.llm = self._init_llm()
        self.agent = self._init_agent()
        self.response_cache = {}
        self.last_query = None

    def _init_elasticsearch(self):
        """Initialisation de la connexion Elasticsearch"""
        try:
            es = Elasticsearch(
                hosts=["https://52be-41-82-223-79.ngrok-free.app"],
                basic_auth=(os.getenv("ELASTIC_USER"), os.getenv("ELASTIC_PASSWORD")),
                verify_certs=False,
                request_timeout=45
            )
            
            if not es.ping():
                raise ConnectionError("❌ Impossible de se connecter à Elasticsearch")
            

            
            if not es.indices.exists(index="assistant_fiscal_v2"):
                print("⚠️ L'index 'assistant_fiscal_v2' n'existe pas. Créez-le avec le mapping approprié.")

            print("✅ Connexion Elasticsearch établie")

            return es
            
        except Exception as e:
            print(f"⚠️ Erreur Elasticsearch : {e}")
            return None

    def _init_embedder(self):
        """Chargement du modèle d'embedding"""
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def _init_llm(self):
        """Configuration du LLM"""
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1500
        )

    def _get_contextual_results(self, query: str) -> Tuple[List[str], float]:
        """Recherche optimisée dans Elasticsearch"""
        try:
            res = self.es.search(
                index="assistant_fiscal_v2",
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["question^3", "reponse^2", "tags"],
                                        "type": "best_fields"
                                    }
                                }
                            ]
                        }
                    },
                    "size": 3
                }
            )

            hits = res.get('hits', {}).get('hits', [])
            if not hits:
                return [], 0

            best_score = hits[0]['_score']
            responses = [hit['_source']['reponse'] for hit in hits]
            
            print("\n🔍 Résultats de recherche :")
            for i, hit in enumerate(hits[:3]):
                print(f"{i+1}. Score: {hit['_score']:.2f} | Question: {hit['_source']['question']}")
            
            return responses, best_score

        except Exception as e:
            print(f"⚠️ Erreur recherche Elasticsearch: {e}")
            return [], 0

    def _gerer_salutation(self):
        """Gestion simplifiée des salutations"""
        return "💼 Bonjour ! Assistant fiscal sénégalais à votre service. Posez-moi vos questions sur les impôts et taxes."

    def _est_question_fiscale(self, query: str) -> bool:
        """Détermine si la question est de nature fiscale"""
        query_lower = query.lower()
        salutations = {"bonjour", "salut", "hello", "bonsoir", "coucou", "hi", "salam"}
        
        if any(salut in query_lower for salut in salutations):
            return False
            
        return any(mot in query_lower for mot in self.mots_cles_fiscaux)

    def recherche_fiscale(self, query: str) -> str:
        """Version stricte qui ne répond qu'aux questions fiscales"""
        print(f"🎯 Appel à recherche_fiscale avec : {query}")

        # Étape 1 : Détection de la langue
        try:
            langue = detect(query)
            if langue != "fr":
                return "⛔ Veuillez poser votre question en français uniquement."
        except LangDetectException:
            return "⚠️ Impossible de détecter la langue de votre question."

        # Étape 2 : Gestion des salutations
        if not self._est_question_fiscale(query):
            return (
                "⛔ Je suis strictement limité aux questions fiscales sénégalaises. "
                "Posez-moi une question sur les impôts, taxes, déclarations fiscales, etc. "
                "Pour plus d'informations, consultez : https://www.dgid.sn/procedures-fiscales/"
            )

        # Étape 3 : Recherche contextuelle uniquement pour les questions fiscales
        responses, score = self._get_contextual_results(query)
        
        if responses:
            return responses[0]
        else:
            return (
                "🔍 Je n'ai pas trouvé de réponse précise dans ma base de données fiscales. "
                "Consultez le site officiel : https://www.dgid.sn/procedures-fiscales/"
            )

    def vider_cache(self):
        """Vide le cache des réponses"""
        self.response_cache.clear()
        print("🗑️ Cache vidé avec succès !")

    def _init_agent(self):
        """Initialisation de l'agent LangChain avec un outil fiscal strict"""
        fiscal_tool = Tool(
            name="BaseFiscalePremium",
            func=self.recherche_fiscale,
            description=(
                "🔍 Outil strictement limité à la fiscalité sénégalaise. "
                "Répondez uniquement en français, meme si lE LLM DONNE UNE REPONSE EN ANGLAIS TRADUIT LA EN français. "
                "Ne répond qu'aux questions sur les impôts, taxes, déclarations, "
                "procédures fiscales et textes de loi. Renvoie systématiquement vers "
                "https://www.dgid.sn/procedures-fiscales/ pour les questions non fiscales."
            )
        )

        return initialize_agent(
            tools=[fiscal_tool],
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='output'
            ),
            verbose=False,
            max_iterations=4,
            early_stopping_method="generate",
            agent_kwargs={
               "system_message": SystemMessage(content="""
🎓 Vous êtes un expert fiscal sénégalais STRICT, spécialisé uniquement dans le droit fiscal.

RÈGLES ABSOLUES :
1. NE RÉPONDEZ QU'AUX QUESTIONS FISCALES
2. Pour toute question non fiscale, répondez :
   "⛔ Désolé, je suis strictement limité aux questions fiscales sénégalaises. Consultez https://www.dgid.sn/procedures-fiscales/"
3. Structurez toujours vos réponses en :
   - Contexte fiscal
   - Points clés (3-5 maximum)
   - Référence légale si disponible
   - Lien vers DGID si pertinent
   - Répondez UNIQUEMENT en français
   - Pour les salutations comme Bonjour, Salut, hi, répondez "Bonjour ! Je suis votre assistant fiscal sénégalais. Posez-moi vos questions sur les impôts et taxes."
EXEMPLES DE RÉPONSES :

✅ "Quelles sont les obligations fiscales pour créer une entreprise ?"
→ "Pour créer une entreprise au Sénégal, voici les aspects fiscaux :
   1. Immatriculation au registre fiscal (NINEA)
   2. Déclaration d'existence sous 30 jours
   3. Paiement de la patente
   Réf: Code général des impôts, Article 12. Plus d'infos: https://www.dgid.sn"

NE FAITES PAS :
- De suppositions hors domaine fiscal
- De réponses créatives non fiscales
- D'interprétations personnelles
-Ne r2pondez JAMAIS en anglais
""")
            }
        )

    def run(self):
        """Lancement de l'interface utilisateur"""
        print("\n" + "="*50)
        print("ASSISTANT FISCAL PREMIUM - SÉNÉGAL ".center(50))
        print("="*50)
        print(self._gerer_salutation())
        
        while True:
            try:
                user_input = input("\nVotre question fiscale : ").strip()
                
                if user_input.lower() in ['au revoir', 'merci', 'quit', 'q']:
                    print("\nMerci pour votre confiance. À bientôt !")
                    break
                
                if user_input.lower() in ['bonjour', 'salut', 'salam', 'hi']:
                    print("\nBonjour ! Je suis votre assistant fiscal sénégalais. Posez-moi vos questions sur la fiscalité.")
                    continue
                   
                    
                if user_input.lower() in ['vider cache', 'reset']:
                    self.vider_cache()
                    continue
                    
                print("\n🔍 Consultation de la base fiscale...")
                response = self.agent.invoke({"input": user_input})
                print("\n📌 Réponse :", response['output'])
                
            except KeyboardInterrupt:
                print("\n\nMerci d'avoir utilisé l'Assistant Fiscal Premium. Au revoir !")
                break
                
            except Exception as e:
                print(f"\n⚠️ Une erreur est survenue : {str(e)}")
                print("Veuillez reformuler votre question ou contacter le support technique.")
                self.vider_cache()

if __name__ == "__main__":
    assistant = PremiumFiscalAssistant()
    assistant.run()
