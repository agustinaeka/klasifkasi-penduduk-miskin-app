import mysql.connector

db = mysql.connector.connect(
    user='root',
    password='',
    host='localhost',
    database='magang')

mycursor = db.cursor()


def adduser(username, password):
    sql = "INSERT INTO login(username,password) VALUES (%s, %s)"
    val = (username, password)
    mycursor.execute(sql, val)
    db.commit()


def login_user(username, password):
    sql = "SELECT * FROM login WHERE username=(%s) AND password =(%s)"
    val = (username, password)
    mycursor.execute(sql, val)
    data = mycursor.fetchall()
    return data
