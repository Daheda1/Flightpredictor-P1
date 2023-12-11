import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings
import joblib
warnings.filterwarnings("ignore", category=FutureWarning)

def get_dataset(relevant_columns, nrows):
        
    #Henter vores datasæt og laver det til pandas dataframe
    df = pd.read_csv('Combined_Flights_2022.csv', usecols=relevant_columns + ['ArrDelayMinutes'], nrows = nrows)

    #DelayLabel bliver tilføjet og apply bruger funktionen label_delay på hele rækken
    df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)

    df.dropna(inplace=True)

    return df.pop("DelayLabel"), df

def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 120:
        return 'late'
    else:
        return 'very-late'
    
def ready_dataset(relevant_columns, categorical_columns, continuous_columns, nrows):
    label, df = get_dataset(relevant_columns, nrows)
    print("Datasæt indlæst")

    train_x, test_x, train_y, test_y = train_test_split(df, label, stratify=label, test_size=0.20, random_state=1)
    print("80/20 Split lavet")

    print("Påbegynder SMOTE")
    smote_nc = SMOTENC(categorical_features = categorical_columns,random_state=42)
    train_x, train_y = smote_nc.fit_resample(train_x, train_y)
    print("SMOTE Fuldført")

    print("Påbegynder One-Hot")
    train_x = pd.get_dummies(train_x, columns=categorical_columns, dtype=int, sparse=True)
    test_x = pd.get_dummies(test_x, columns=categorical_columns, dtype=int, sparse=True)
    print("One-Hot Fuldført")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x[continuous_columns])
    test_x = scaler.fit_transform(test_x[continuous_columns])


    return train_x, test_x, train_y, test_y

def Train_and_save_models(train_x, test_x, train_y, test_y, models):
    for model in models:
        # Model name
        model_name = model.__class__.__name__
        print(f"Running cross-validation for {model_name}")

        # Cross-validation
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = {}

        for metric in scoring_metrics:
            # Perform cross-validation for each metric
            cv_scores = cross_val_score(model, train_x, train_y, cv=5, scoring=metric)
            print(cv_scores)
    
            # Store the cross-validation scores for each metric
            cv_results[metric] = cv_scores

        # Train model on the entire dataset
        model.fit(train_x, train_y)

        # Save the trained model
        joblib.dump(model, f'{model_name}.joblib')

        # Evaluate the model
        scores = evaluate_model(model, test_x, test_y)

        # Skriv både cross-validation og evaluering scores til samme fil
        with open(f"{model_name}_performance.txt", "w") as f:
            f.write(f"Model: {model_name}\n")

            # Skriv cross-validation scores
            f.write("Cross-validation results:\n")
            for metric, cv_score in cv_results.items():
                f.write(f"{metric}: {cv_score.tolist()}\n")
                f.write(f"Mean {metric}: {np.mean(cv_score)}\n")
            f.write("\n")

            # Skriv evaluering scores
            f.write("Evaluation results:\n")
            for metric, score in scores.items():
                f.write(f"{metric}: {np.mean(score) if isinstance(score, list) else score}\n")

def calculate_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity.append(spec)
    return np.mean(specificity)
    
def evaluate_model(model, X_test, y_test):
    labels = sorted(y_test.unique())
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, labels=labels, average='macro'),
        "recall": recall_score(y_test, y_pred, labels=labels, average='macro'),
        "f1_score": f1_score(y_test, y_pred, labels=labels, average='macro'),
        "specificity": calculate_specificity(y_test, y_pred, labels)
    }    
    return scores
    
def main():
    #Definere de kolonner vi gerne vil træne på
    nrows = 500000

    relevant_columns = ['Airline', 'Origin', 'Dest', 
                        'DepTime', 'ArrTime', 'Distance', 
                        'DayOfWeek', 'DayofMonth', 'Quarter']
    
    categorical_columns = ['Airline', 'Origin', 'Dest']
    continuous_columns = ["DepTime", "ArrTime", 'Distance']

    # Initialisering af modeller
    rfc = RandomForestClassifier(random_state=42)
    dtc = DecisionTreeClassifier(random_state=42)
    gnb = GaussianNB()
    knc = KNeighborsClassifier()

    # Liste af modeller til krydsvalidering og gemning
    models = [rfc, dtc, gnb, knc]       

    train_x, test_x, train_y, test_y = ready_dataset(relevant_columns, categorical_columns, continuous_columns, nrows)

    Train_and_save_models(train_x, test_x, train_y, test_y, models)


main()