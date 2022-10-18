#open text file in read mode
text_file = open("500x200_rbf.libsvm.model", "r")
 
#read whole file to a string
data = text_file.read()
 
#close file
text_file.close()
 

for i in range(199, -1, -1):
  data = data.replace(" {}:".format(i), " {}:".format(i + 1))


with open('500x200_rbf.libsvm.model', 'w', encoding='utf-8') as f:
    f.write(data)
