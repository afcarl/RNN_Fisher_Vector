out = open("output.txt", 'w') # change name
fo = open("input.txt", 'r')
for line in fo:
    if line != "\n":
        out.write(line)
        out.write("\n")
        
out.close()