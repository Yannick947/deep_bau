#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:53:55 2021

@author: tquentel
"""
import os
import json
import requests as rq

api_dict = {"stammdaten": ["baubereiche",
                           "baustellen",
                           "bauteile",
                           "geraete",
                           "material",
                           "materialgruppen",
                           "geraetetypen",
                           "notizkategorien",
                           "personaltypen",
                           "taetigkeiten",
                           "teams",
                           "wetterarten",
                           "formbuilders"
                           ],
            "erfassungsdaten": ["personalzeiten",
                                "geraete",
                                "leistungen",
                                "material",
                                "notizen",
                                "wetter",
                                "forms"
                                ],
            "lohndaten": ["feiertage",
                          "krankheitstage",
                          "sonstiges",
                          "urlaub"
                          ]
            }

user = os.environ['sdacathon_user']
pwd = os.environ['sdacathon_pwd']

missed = []
for sub, topics in api_dict.items():
    for topic in topics:

        link = f"https://server.123erfasst.de/api/{sub}/{topic}?referby=ID"

        try:
            data = json.loads(
                rq.get(link, auth=rq.auth.HTTPBasicAuth(user, pwd)).text)

            df = pd.json_normalize(data)
            df.to_csv(f"{sub}/{topic}.csv")

        except JsonError:
            missed.append(topic)
