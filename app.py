import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")

@app.route('/minor')
def minor():
  
    return render_template("minor.html")

@app.route('/major')
def major():
  
    return render_template("major.html")

@app.route('/gallery')
def gallery():
  
    return render_template("gallery.html")

@app.route('/contact')
def contact():
  
    return render_template("contact.html")
  
@app.route('/resume')
def resume():
  
    return render_template("resume.html")

@app.route('/decision')
def model1():
  
    return render_template("decision.html")

@app.route('/logistic')
def model2():
  
    return render_template("logistic.html")

@app.route('/svm')
def model3():
  
    return render_template("svm.html")

@app.route('/random')
def model4():
  
    return render_template("random.html")

@app.route('/knn')
def model5():
  
    return render_template("knn.html")

@app.route('/naive')
def model6():
  
    return render_template("naive.html")




@app.route('/predict_decision',methods=['GET'])
def predict1():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('decision_model_major.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
      message="Not Healithy"
    elif prediction==1:
      message="Healithy"
    
        
    return render_template('decision.html', prediction_text='{}'.format(message))


@app.route('/predict_logistic',methods=['GET'])
def predict2():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('major_log_reg.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
     
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
       message="Not Healithy"
    elif prediction==1:
       message="Healithy"
     
    
        
    return render_template('logistic.html', prediction_text='{}'.format(message))

@app.route('/predict_svm',methods=['GET'])
def predict3():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('svm_major.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
      message="Not Healithy"
    elif prediction==1:
      message="Healithy"
    
        
    return render_template('svm.html', prediction_text='{}'.format(message))

@app.route('/predict_random',methods=['GET'])
def predict4():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('random_forest_major.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
      message="Not Healithy"
    elif prediction==1:
      message="Healithy"
    
        
    return render_template('random.html', prediction_text='{}'.format(message))

@app.route('/predict_knn',methods=['GET'])
def predict5():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('knn_major.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
      message="Not Healithy"
    elif prediction==1:
      message="Healithy"
    
        
    return render_template('knn.html', prediction_text='{}'.format(message))

@app.route('/predict_naive',methods=['GET'])
def predict6():
    
    
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.args.get('age'))
    sex=float(request.args.get('sex'))
    cp=float(request.args.get('cp'))
    trestbps=float(request.args.get('trestbps'))
    chol=float(request.args.get('chol'))
    fbs=float(request.args.get('fbs'))
    restecg=float(request.args.get('restecg'))
    thalach=float(request.args.get('thalach'))
    exang = float(request.args.get('exang'))
    oldpeak=float(request.args.get('oldpeak'))
    slope=float(request.args.get('slope'))
    ca=float(request.args.get('ca'))
    thal=float(request.args.get('thal'))
   
    model=pickle.load(open('naive_major.pkl','rb'))
      

    dataset= pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    
    prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction==0:
      message="Not Healithy"
    elif prediction==1:
      message="Healithy"
    
        
    return render_template('naive.html', prediction_text='{}'.format(message))


app.run()
# if __name__ == "__main__":
    # app.run(debug=True)
