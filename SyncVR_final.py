import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json


df = pd.read_csv('dataset.csv')
df

### Make all values lowercase, remove leading and trailing spaces, and replace underscores with spaces

def replace_space_and_underscore(df):
    unique_values = set()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip().replace("_", " ") if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: x[1:].replace("", "") if isinstance(x, str) and x.startswith(" ") else x)
        df[col] = df[col].apply(lambda x: x.replace("  ", " ") if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        if col != df.columns[0]:
            unique_values.update(set(df[col].tolist()))
    df = pd.DataFrame(df)
    unique_values = [x for x in unique_values if str(x) != 'nan']
    return list(unique_values), df


unique_values = replace_space_and_underscore(df)[0]

len(unique_values)

unique_values

cleaned_df = replace_space_and_underscore(df)[1]


#### Create a new dataframe with the unique values as columns

new_df = pd.DataFrame(columns=unique_values)


### One-hot encode the symptoms into new_df

for col in cleaned_df.columns:
    if col != cleaned_df.columns[0]:
        for value in cleaned_df[col].unique():
            new_df[value] = cleaned_df[col].apply(lambda x: 1 if x == value else 0)


encoded_df = pd.concat([cleaned_df.iloc[:, 0], new_df], axis=1)
encoded_df
encoded_df.to_csv('encoded_encoded_df.csv', index=False)


### Drop the NaN column
encoded_df = encoded_df.iloc[:, :-1]
encoded_df


### Randomize the order of the rows in the dataframe
encoded_df = encoded_df.sample(frac=1)

# Reset the index of the randomized dataframe
encoded_df = encoded_df.reset_index(drop=True)
encoded_df


# Extract the disease column
diseases = encoded_df['Disease']

# Calculate the frequency of each disease
disease_counts = diseases.value_counts()

# Sort the diseases by frequency in descending order
disease_counts_sorted = disease_counts.sort_values(ascending=False)

# Print summary statistics for the disease counts
print("Summary Statistics for Disease Counts:\n")
print(disease_counts_sorted.describe().to_string(float_format='%.1f'))
print("\n")

# Extract the symptom columns
symptoms = encoded_df.iloc[:, 0]

# Calculate the frequency of each symptom
symptom_counts = symptoms.sum(axis=0)

# Sort the symptoms by frequency in descending order
symptom_counts_sorted = symptom_counts.sort_values(ascending=False)

# Print summary statistics for the symptom counts
print("Summary Statistics for Symptom Counts:\n")
print(symptom_counts_sorted.describe().to_string(float_format='%.1f'))


# Calculate the frequency of each symptom
symptom_counts = encoded_df.iloc[:, 1:].sum(axis=0)

# Print the summary statistics
print("Summary Statistics:")
print(symptom_counts.describe().to_string(float_format='%.1f'))

# Calculate the frequency of each symptom
symptom_counts = encoded_df.iloc[:, 1:].sum(axis=0)

# Find the symptom with the lowest count
min_count = symptom_counts.min()
min_count_symptoms = list(symptom_counts[symptom_counts == min_count].index)

min_count_symptoms

# Print the result
print("Symptom(s) with the lowest count:", ', '.join(min_count_symptoms))

# Find the symptom with the highest count
max_count_symptom = symptom_counts.idxmax()
print("Symptom with the highest count:", max_count_symptom)


# Calculate the frequency of each symptom
symptom_counts = encoded_df.iloc[:, 1:].sum(axis=0)

# Create a box and whisker plot of the symptom counts with wider boxes
fig, ax = plt.subplots()
ax.boxplot(symptom_counts, widths=0.5)
ax.set_xlabel('Symptom')
ax.set_ylabel('Frequency')
ax.set_title('Symptom Counts')
plt.show()


### Create a KNN model for the encoded_df
X = encoded_df.iloc[:, 1:]
y = encoded_df.iloc[:, 0]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=9, p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)



# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)

# Calculate the accuracy for each disease
accuracy_df = pd.DataFrame({'Disease': y_test, 'Prediction': y_pred})
accuracy_df['Correct'] = accuracy_df['Disease'] == accuracy_df['Prediction']
accuracy_by_disease = accuracy_df.groupby('Disease')['Correct'].mean().reset_index()

# Create a bar plot of the accuracy for each disease
plt.figure(figsize=(10, 8))
plt.bar(x=accuracy_by_disease['Disease'], height=accuracy_by_disease['Correct']*100)
plt.title("Accuracy by Disease - Overall Accuracy: {:.1%}".format(accuracy), size=16)
plt.xlabel("Disease", size=14)
plt.ylabel("Accuracy (%)", size=14)
plt.ylim([0, 100])
plt.xticks(rotation=90)

### Add a horizontal line for the overall accuracy
plt.axhline(y=accuracy*100, color='r', linestyle='-')

## Add a legend to the top right of the plot
plt.legend(['Average Accuracy', 'Accuracy by Disease'], loc='upper right')
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
plt.show()


# Create a dictionary that maps the numeric labels to disease names
disease_dict = dict(enumerate(encoded_df.iloc[:, 0].unique()))
print(json.dumps(disease_dict, indent=4))
