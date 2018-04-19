import csv

files = {1:"2005_2006_Bruins.csv",2:'2006_2007_Bruins.csv',3:'2007_2008_Bruins.csv',4:'2008_2009_Bruins.csv',5:'2009_2010_Bruins.csv'}
count = 1
for i in range(1,6):
    with open(files[i],'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        with open('Bruins_2005_2010.csv','a') as new_file:
            csv_writer = csv.writer(new_file,delimiter=',')

            wins = {}
            loss = {}
            ot = {}
            home = {}
            away = {}
            header = "gn,gp,date,place,team,gf,ga,outcome,ot,totW,totL,totOT,streak,bSh,bPM,bPPG,bSHG"
            csv_writer.writerow(header)
            for line in csv_reader:
                line[0] = count
                ##Setting home or away tap to .25 or -.25 weights
                if line[3] =='@':
                    line[3] = -.25
                    away[line[1]] = "Away"
                else:
                    line[3] = .25
                    home[line[1]] = 'Home'

                streak = str(line[12])
                # if len(streak) ==3:
                #     if streak[0] == 'W':



                ##Setting wins and losses to 1 or -1 or -.5 for ot or SO loss
                if line[7] == 'W':
                    line[7]=1
                    wins[line[1]]=line[7]
                elif line[7] == 'L':
                    if line[8] =='OT' or line[8] =='SO':
                        line[7] = -.5
                        ot[line[1]] = line[8]
                    else:
                        line[7] = -1
                        loss[line[1]]=line[7]

                csv_writer.writerow(line)
                count += 1
            #print len(ot)+len(wins)+len(loss)
            #print line[3]
            print line[1]



