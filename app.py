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

# Désactivation des avertissements
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
load_dotenv()

class PremiumFiscalAssistant:
    def __init__(self):
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
                hosts=[os.getenv("ELASTIC_ENDPOINT")],
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
            for i, hit in enumerate(hits[:3]):  # Affiche les 3 premiers résultats
                print(f"{i+1}. Score: {hit['_score']:.2f} | Question: {hit['_source']['question']}")
            
            return responses, best_score

        except Exception as e:
            print(f"⚠️ Erreur recherche Elasticsearch: {e}")
            return [], 0

    def _gerer_salutation(self):
        """Gestion simplifiée des salutations sans date"""
        return "💼 Bonjour ! Assistant fiscal sénégalais à votre service. Posez-moi vos questions sur les impôts et taxes."

    def recherche_fiscale(self, query: str) -> str:
        """Version stricte qui ne répond qu'aux questions fiscales"""
        mots_cles_fiscaux = [
            "impôt","impot", "taxe", "TVA", "CFPNB","cfpnb","pv","PV","PME","quitus" , "PCF","fiscalité", "déclaration","CGU","Patente","récapitulatifs",
            "exonération", "remboursement", "trop perçu", "délai","quitus fiscal ", "délai de paiement","quittance","récépissé","revenus","formalisation",
            "contribution", "taxation", "droit d'enregistrement", "droits d'enregistrement", "taxes d'enregistrement","entreprise",'changement de statuts',
            "taxes sur les salaires", "taxe sur les salaires", "taxe foncière", "taxe professionnelle","NINEA","direct","indirect","réouverture"
            "taxe sur la valeur ajoutée", "tva", "passeport", "taxe sur les boissons","réductions","immatriculation","propriétaire","compte","duplicata",
            "IR", "IS", "patente", "douane", "régime fiscal", "code général des impôts","procédure","acte administratif","exonérations",
            "obligation fiscale", "penalité", "amende", "contrôle fiscal","démarrage des activités","homologation","acte"
            "titre","SIGTAS","imposition","bail","foncier bâti ","foncier non bâti","TEOM","vvérification"
        ]
        salutations = ["bonjour", "salut", "hello", "bonsoir", "coucou", "hi", "salam"]
        
        query_lower = query.lower()
        
        # Gestion des salutations
        if any(salut in query_lower for salut in salutations):
            return self._gerer_salutation()
        
        # Vérification stricte du domaine fiscal
        if not any(mot in query_lower for mot in mots_cles_fiscaux):
            return ("⚠️ Je suis un assistant spécialisé exclusivement en fiscalité sénégalaise.\n")
        # Recherche dans Elasticsearch
        responses, score = self._get_contextual_results(query)
        
        if not responses:
            return ("🔍 Je n'ai pas trouvé d'information précise dans ma base fiscale. "
                "Consultez le site officiel : www.impotsetdomaines.gouv.sn\n")

        return responses[0]

    def vider_cache(self):
        """Vide le cache des réponses"""
        self.response_cache.clear()
        print("🗑️ Cache vidé avec succès !")

    def _init_agent(self):
        """Initialisation de l'agent LangChain"""
        fiscal_tool = Tool(
            name="BaseFiscalePremium",
            func=self.recherche_fiscale,
            description="Base de connaissances sur la fiscalité sénégalaise"
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
Vous êtes un expert fiscal sénégalais. Répondez de manière précise et structurée :
1. Donnez des réponses factuelles basées sur la réglementation
2. Citez vos sources quand c'est possible
3. Pour les questions hors sujet, redirigez vers www.impotsetdomaines.gouv.sn
                """)
            }
        )

    def run(self):
        """Lancement de l'interface utilisateur avec gestion avancée des salutations"""
        # Message d'accueil initial
        print("\n" + "="*50)
        print("ASSISTANT FISCAL PREMIUM - SÉNÉGAL ".center(50))
        print("="*50)
        print(self._gerer_salutation())  # Affiche le message d'accueil initial
        
        while True:
            try:
                user_input = input("\nVotre question fiscale : ").strip()
                
                # Commandes spéciales
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nMerci pour votre confiance. À bientôt !")
                    break
                    
                if user_input.lower() in ['vider cache', 'reset']:
                    self.vider_cache()
                    print("🗑️ Cache vidé avec succès !")
                    continue
                    
                # Détection des salutations en cours de session
                salutations = ["bonjour", "salut", "hello", "bonsoir", "coucou", "hi", "salam"]
                if any(salut in user_input.lower() for salut in salutations):
                    print("\n" + self._gerer_salutation())
                    continue
                    
                # Traitement des questions fiscales
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
