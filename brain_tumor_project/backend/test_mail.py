import smtplib

EMAIL = "vedaharsha1811@gmail.com"
PASSWORD = "hieuaukhqbsygfnj"

server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
server.login(EMAIL, PASSWORD)
print("LOGIN SUCCESS")
server.quit()