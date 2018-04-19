import csv

files = {1:"2005_2006_Bruins copy.csv",2:'2006_2007_Bruins copy.csv'}
count = 1
for i in range(1,3):
    with open(files[i],'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        with open('Bruins_2005_2007.csv','a') as new_file:
            csv_writer = csv.writer(new_file,delimiter=',')
            header = 'team,HA,SF,PMF,PPGF,SHGF,SA,PMA,PPGA,SHGA'
            csv_writer.writerow(header)
            for line in csv_reader:
                if line[1] =='@':
                    line[1] = 0
                else:
                    line[1] = 1


                if line[2] == 'W':
                    line[2]=1
                elif line[2] == 'L':
                    line[2] = 0
                csv_writer.writerow(line)
                count += 1

            print line



