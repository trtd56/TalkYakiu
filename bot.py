# coding: utf-8
import json
import os
import requests
import falcon


class CallbackResource(object):

    def on_post(self, req, resp):
        REPLY_ENDPOINT = 'https://api.line.me/v2/bot/message/reply'
        API = os.environ["API"]
        KEY = "Bearer " + os.environ["KEY"]
        header = {"Content-Type": "application/json",
                  "Authorization": KEY}

        body = req.stream.read()
        receive_params = json.loads(body.decode('utf-8'))

        reply_token = receive_params["events"][0]["replyToken"]
        text = receive_params["events"][0]["message"]["text"]

        req_str = "http://"+API+":5000/talk/"+text
        print(req_str)
        yakiu = requests.get(req_str)
        yakiu = yakiu.json()
        print(yakiu)
        res_yakiu = yakiu["ResultSet"]["Yakiu"]
        unk_yakiu = "unk:\n" + yakiu["ResultSet"]["unk"]

        payload = {"replyToken":reply_token,
                   "messages":[{"type":"text",
                                "text":res_yakiu},
                               {"type":"text",
                                "text":unk_yakiu}]}
        payload = json.dumps(payload)

        requests.post(REPLY_ENDPOINT, headers=header, data=payload)
        resp.body = json.dumps('OK')

api = falcon.API()
api.add_route('/callback', CallbackResource())
