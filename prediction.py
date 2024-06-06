import pandas as pd

from sklearn.impute import SimpleImputer

def predict_new_film(new_film, vectorizer, model, imputer=None):
    # Préparer les données du nouveau film
    new_film['genres'] = ' '.join(new_film['genres'])
    new_film['keywords'] = ' '.join(new_film['keywords'])
    new_film['production_companies'] = ' '.join(new_film['production_companies'])
    new_film['production_countries'] = ' '.join(new_film['production_countries'])
    new_film['spoken_languages'] = ' '.join(new_film['spoken_languages'])
    
    new_film_text = new_film['original_title'] + ' ' + \
                    new_film['genres'] + ' ' + \
                    new_film['keywords'] + ' ' + \
                    new_film['production_companies'] + ' ' + \
                    new_film['production_countries'] + ' ' + \
                    new_film['spoken_languages']
    
    new_film_tfidf = vectorizer.transform([new_film_text])
    new_film_tfidf_df = pd.DataFrame(new_film_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    
    new_film_features = pd.DataFrame({
        'budget': [new_film['budget']],
        'runtime': [new_film['runtime']]
    })
    
    new_film_features = pd.concat([new_film_features.reset_index(drop=True), new_film_tfidf_df], axis=1)
    
    if imputer is not None:
        new_film_features = imputer.transform(new_film_features)
    
    # Utiliser le modèle entraîné pour prédire la note
    predicted_vote_average = model.predict(new_film_features)
    
    return predicted_vote_average[0]
