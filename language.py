
LANGUAGES = {"it": "Italian", "pl": "Polish", "ru": "Russian", "sk": "Slovak", "pt": "Portuguese",
             "ro": "Romanian", "da": "Danish", "sv": "Swedish", "no": "Norwegian", "en": "English",
             "es": "Spanish", "fr": "French", "cs": "Czech", "de": "German", "fi": "Finnish", "et": "Estonian",
             "lv": "Latvian", "lt": "Lithuanian", "fa": "Persian", "hu": "Hungarian", "he": "Hebrew",
             "el": "Greek", "ar": "Arabic","tl":"Austronesian","Japense":"ja","Kashmiri":"ks","Malyalam":"ml","Marathi":"mr","Nepali":"ne","Sindhi":"sd"}

def input_line(line):
    from  langdetect import detect
    for i,j in zip(LANGUAGES.keys(),LANGUAGES.values()):
        if detect(line) in i:
            print("THE LANGUAGE IS\n"," ",j)

text=input("ENTER ANY TEXT\n")
input_line(text)





