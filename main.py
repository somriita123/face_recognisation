import cv2
import os, shutil
from flask import Flask, request, render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 100

imgBackground=cv2.imread("background.png")

print_log = []
#print_log.append('isdir')
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    #print("=============>>>>>", userid)

    if userid.isdigit():
        try:
            if int(userid) not in list(df['Roll']):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
        except ValueError:
            # do something else
            print("userid is not valid.")
    

def getallusers():    
    all_users = os.listdir('static/faces')
    allusers = len(all_users)
    faces = []
    faces_pics = []

    for user in all_users:
        faces.append(user)
        imagename = os.listdir(f'static/faces/{user}')
        if imagename and imagename[0]:
            faces_pics.append(f'{user}/{imagename[0]}')

    return allusers, faces, faces_pics

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    allusers, faces, faces_pics = getallusers()
    print(faces)
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, allusers=allusers, faces=faces, faces_pics=faces_pics, datetoday2=datetoday2)

@app.route('/take-attendance', methods=['GET'])
def takeAttendance():
    names, rolls, times, l = extract_attendance()
    allusers, faces, faces_pics = getallusers()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, allusers=allusers, faces=faces, faces_pics=faces_pics, datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            try:
                faces = extract_faces(frame)
                for (x,y,w,h) in faces:
                    #(x, y, w, h) = extract_faces(frame)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                    face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    identified_person = identify_face(face.reshape(1, -1))[0]
                    #add_attendance(identified_person)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                    cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            except Exception as e:
                print('Failed to get extract_faces%s. Reason: %s' % e)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        try:
            cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print('Failed to set porperty%s. Reason: %s' % e)
        
        #PRESS 'C' TO CLOSE WINDOW
        if cv2.waitKey(1) == 99 or cv2.waitKey(1) == 67:
            cap.release()
            cv2.destroyAllWindows()
            train_model()
            break
        #PRESS '0' TO TAKE ATTENDANCE
        if cv2.waitKey(1) == 48 and identified_person:
            add_attendance(identified_person)
            cap.release()
            cv2.destroyAllWindows()
            train_model()
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    # names, rolls, times, l = extract_attendance()
    # allusers, faces, faces_pics = getallusers()
    # return render_template('home.html', names=names, rolls=rolls, times=times, l=l, allusers=allusers, faces=faces, faces_pics=faces_pics, datetoday2=datetoday2)
    return redirect(url_for('home'))

@app.route('/remove-attendance/<int:row_id>')
def removeAttendance(row_id):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    df = df.drop(df.index[row_id])
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
    return redirect(url_for('home'))

@app.route('/remove-faces/<user>')
def removeFaces(user):    
    if user:
        for filename in os.listdir(f'static/faces/{user}'):
            file_path = os.path.join(f'static/faces/{user}', filename)

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        try:
            userfolder = os.path.join(f'static/faces/', user)
            if os.path.exists(userfolder) and os.path.isdir(userfolder):
                shutil.rmtree(userfolder)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (userfolder), e)

    return redirect(url_for('home'))


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding New User', frame)
        try:
            cv2.setWindowProperty('Adding New User', cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print('Failed')
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    #print('Training Model')
    train_model()
    #names, rolls, times, l = extract_attendance()
    #allusers, faces, faces_pics = getallusers()
    #return render_template('home.html', names=names, rolls=rolls, times=times, l=l, allusers=allusers, faces=faces, faces_pics=faces_pics, datetoday2=datetoday2)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)



#py -3 main.py