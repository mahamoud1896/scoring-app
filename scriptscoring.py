import pandas as pd
import streamlit as st
import joblib  # Pour charger le modèle
from sklearn.metrics import precision_score
#from annotated_text import annotated_text



# Chargement du modèle random forestet svm formé

#model_rf = joblib.load('final_rf16_model.pkl')

model_svm = joblib.load('final_svm16_model.pkl')


# fonction pour prédire la solvabilité du client avec le modèle random forest
def predict_solvency_rf(solde_compte_courant, duree_credit, montant_impaye, montant_credit,
                                  age, solde_compte_epargne, profile, Retard_P, Presence_Remboursement_Anticipe,
                                     GENRE, Nombre_de_Credits, seuil=0.5):
    
    """
    Prédiction de la solvabilité du client avec le modèle Random Forest.

    :param solde_compte_courant: Solde du compte courant
    :param duree_credit: Durée du crédit en mois
    :param montant_impaye: Montant impayé
    :param montant_credit: Montant du crédit
    :param age: Âge du client
    :param solde_compte_epargne: Solde du compte épargne
    :param profile: Profil du client
    :param statut: Statut du crédit
    :param nombre_credits: Nombre de crédits
    :param nombre_de_compte: Nombre de comptes
    :param pas_de_remboursmt_anticip: Pas de remboursement anticipé (0 ou 1)
    :param retard_paiement: Retard de paiement en jours
    :param genre_encode: Genre du client (0 pour Féminin, 1 pour Masculin)
    :return: Résultat de la prédiction ("Client SOLVABLE" ou "Client NON SOLVABLE")
    """
    global probabilities
    
    # dataframe avec les paramètres
    data = pd.DataFrame({
        'SOLDE_COMPTE_COURANT': [solde_compte_courant],
        'DUREE_CREDIT': [duree_credit],
        'MONTANT_IMPAYE': [montant_impaye],
        'MONTANT_CREDIT': [montant_credit],
        'AGE': [age],
        'SOLDE_COMPTE_EPARGNE': [solde_compte_epargne],
        'PROFILE': [profile],
        #'STATUT': [statut],
        'Retard_P': [Retard_P],
        'Presence_Remboursement_Anticipe': [Presence_Remboursement_Anticipe],
        'GENRE': [GENRE],
        #'Nombre_Impayes': [Nombre_Impayes],
        'Nombre_de_Credits': [Nombre_de_Credits]
    })

    # la prédiction avec le modèle Random forest
    #probabilities = model_rf.predict_proba(data)[:, 1]  # Probabilité d'être "Client SOLVABLE"

    # Renvoyer le résultat en fonction du seuil
    #if probabilities >= seuil:
    #    return "Client SOLVABLE", probabilities
    #else:
    #    return "Client NON SOLVABLE", probabilities

    




#============================================================================================================

def predict_solvency_svm(solde_compte_courant, duree_credit, montant_impaye, montant_credit,
                        age, solde_compte_epargne, profile, Retard_P, Presence_Remboursement_Anticipe,
                        GENRE, Nombre_de_Credits):
    """
    Prédiction la solvabilité du client avec le modèle SVM.

    :param solde_compte_courant: Solde du compte courant
    :param duree_credit: Durée du crédit en mois
    :param montant_impaye: Montant impayé
    :param montant_credit: Montant du crédit
    :param age: Âge du client
    :param solde_compte_epargne: Solde du compte épargne
    :param profile: Profil du client
    :param statut: Statut du crédit
    :param nombre_credits: Nombre de crédits
    :param nombre_de_compte: Nombre de comptes
    :param pas_de_remboursmt_anticip: Pas de remboursement anticipé (0 ou 1)
    :param retard_paiement: Retard de paiement en jours
    :param genre_encode: Genre du client (0 pour Féminin, 1 pour Masculin)
    :return: Résultat de la prédiction ("Client SOLVABLE" ou "Client NON SOLVABLE")
    """
    global probabilities

    # dataframe avec les paramètres
    data = pd.DataFrame({
        'SOLDE_COMPTE_COURANT': [solde_compte_courant],
        'DUREE_CREDIT': [duree_credit],
        'MONTANT_IMPAYE': [montant_impaye],
        'MONTANT_CREDIT': [montant_credit],
        'AGE': [age],
        'SOLDE_COMPTE_EPARGNE': [solde_compte_epargne],
        'PROFILE': [profile],
        #'STATUT': [statut],
        'Retard_P': [Retard_P],
        'Presence_Remboursement_Anticipe': [Presence_Remboursement_Anticipe],
        'GENRE': [GENRE],
        #'Nombre_Impayes': [Nombre_Impayes],
        'Nombre_de_Credits': [Nombre_de_Credits]
    })

    # Prédiction de la classe
    predicted_class = model_svm.predict(data)

    # Scores de décision
    decision_scores = model_svm.decision_function(data)

    return predicted_class, decision_scores

#==============================================================================================================

# Liste des modèles disponibles
#model_options = ["Random Forest", "SVM"]
#selected_model = st.selectbox("Sélectionnez le modèle", model_options)

st.header("Scoring bancaire")
# Interface utilisateur Streamlit
st.title("Prédiction de Solvabilité ")

st.info("Veuillez saisir ci-dessous les informations du client!")

# Ajout des champs de saisie pour les paramètres
solde_compte_courant = st.number_input("Solde Compte Courant")
duree_credit = st.number_input("Durée Crédit (en mois)")
st.markdown("Montant Impaye | <span style='color: red;'>L'impaye vaut 0 si nouveau client</span>", unsafe_allow_html=True)
# Saisie du montant impayé
montant_impaye = st.number_input("")
montant_credit = st.number_input("Montant Crédit")
age = st.number_input("Âge")
solde_compte_epargne = st.number_input("Solde Compte Épargne")
# Liste des options pour la variable PROFILE
profile_options = [
    "Particuliers REV < 200K",
    "Particuliers REV > 1500K",
    "Particuliers REV 500K << 1500K",
    "Personnel banque",
    "Particuliers à REV moyens",
    "Client interne",
    "Particulier à valider",
    "Diplômé sans emploi",
    "Pas de revenus",
    "Sans emploi sans diplôme"
]
# Sélection du profil par l'utilisateur
selected_profile = st.selectbox("Profile", profile_options)
# Mappe la sélection de l'utilisateur aux valeurs numériques
profile = profile_options.index(selected_profile)

Nombre_de_Credits = st.number_input("Nombre de Crédits")
#Nombre_Impayes = st.number_input("Nombre d'impayes")
#nombre_de_compte = st.number_input("Nombre de Compte")
Presence_Remboursement_Anticipe = st.number_input("Pas de Remboursement Anticipé ")
st.markdown("Retard Paiement (en jours) | <span style='color: red;'>Le retard vaut 0 si nouveau client</span>", unsafe_allow_html=True)
Retard_P = st.number_input("RETARD ")
# Sélection du genre par l'utilisateur
selected_genre = st.selectbox("Genre", ["Féminin", "Masculin"])

# Mappe la sélection de l'utilisateur aux valeurs numériques (0 ou 1)
if selected_genre == "Féminin":
    GENRE = 0
else:  
    GENRE = 1


# Bouton pour lancer la prédiction
if st.button("Prédire"):
    #if selected_model == "Random Forest":
        # Exécute la prédiction avec le modèle random forest
        #result, confidence = predict_solvency_rf(solde_compte_courant, duree_credit, montant_impaye, montant_credit,
         #                         age, solde_compte_epargne, profile, Retard_P, Presence_Remboursement_Anticipe,
         #                            GENRE, Nombre_de_Credits, seuil=0.5
         #                         )
        #st.write("Résultat de la prédiction:", result)
        #st.write("Score de confiance (Probabilité d'être solvable):", confidence)


    #elif selected_model == "SVM":
        # Exécute la prédiction avec le modèle SVM
        results, decision_scores  = predict_solvency_svm(solde_compte_courant, duree_credit, montant_impaye, montant_credit,
                                  age, solde_compte_epargne, profile, Retard_P, Presence_Remboursement_Anticipe,
                                     GENRE, Nombre_de_Credits
                                  )

        st.write("Résultat de la prédiction: ", results)
        st.write("Score de décision: ", decision_scores)




    
   




    
