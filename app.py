from flask import Flask,jsonify,redirect,request,render_template,url_for
import pickle

app=Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict_diabetes():

    with open("Diabetes_model.pkl","rb") as f:
        model_ml=pickle.load(f)

    data = request.form
    print("Data is :",data)

    Glucose = int(data["Glucose"])
    BloodPressure = int(data["BloodPressure"])
    SkinThickness = int(data["SkinThickness"])
    Insulin = int(data["Insulin"])
    BMI = float(data["BMI"])
    DiabetesPedigreeFunction = float(data["DiabetesPedigreeFunction"])
    Age = int(data["Age"])

    result = model_ml.predict([[Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    return jsonify({"Result":f"Based on the personal information,diabetes outcome is {result}"})

if __name__ == "__main__":
    app.run(debug=True)
