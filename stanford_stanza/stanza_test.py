import stanza
stanza.download('hi') # downlo
nlp = stanza.Pipeline('hi') # initialize English neural pipeline
doc = nlp("नोएडा की झुग्गी में 200 लोग कोरोना संदिग्ध, DM बोले- सभी होंगे क्वारनटीन") # run annotation over a sentence
print(doc)