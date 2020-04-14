### deploy app on remote server:
reference: https://github.com/BIU-NLP/lab-wiki/wiki/Running-streamlit-on-the-servers


ssh nlpXX
# check available ports on the server
netstat -tulpn | grep LISTEN
streamlit --log_level debug run app.py --server.port YYYY

edit WWW/.htaccess:
RewriteRule ^app/stream(.*?)$ ws://nlpXX:YYYY/stream$1 [P]
RewriteRule ^app/(.*)$ http://nlpXX:YYYY/$1 [P]

# run inside screen (nlp12)
streamlit --log_level debug run heMoji.py --server.port 1212 |& tee -a ~/emoji_streamlit.log