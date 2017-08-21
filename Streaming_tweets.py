from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import sentiment_mod as s
import time
import json


#consumer key, consumer secret, access token, access secret.
ckey="ZynzmLTK1YjvepzXU0wqw0PA9"
csecret="gkp8eSUHq5zvXTLCD5SjuvP6DFjEzzeFOQxNsdhIU4JIO37a8O"
atoken="899251591966060544-X38RWmofto7RfGHcBNuVuU5OTYMUfKC"
asecret="VMaOHbSHEfnd4jSk38A8lYvwoLwLTkSW0hrSi3CFFQGzh"

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            
            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)
            s_confidence = str(confidence)

            if confidence >= 80:
                output = open("output_files/twitter-out(happy).txt","a")
##                output.write(tweet)
##                output.write(' ')
                output.write(sentiment_value)
##                output.write(' ')
##                output.write(s_confidence)
                output.write('\n')
                output.close()
            
            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
