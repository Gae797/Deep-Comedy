# -*- coding: utf-8 -*-

'''
This class is responsible for preprocessing data for both training and test.
'''

import re

class TextProcessor:
    
    def __init__(self, path):
        
       f = open(path,"r",encoding="utf8")
       self.text = f.read()
       f.close()
       
       self.lines=[]
       
       self.START_LINE_TOKEN = "SL"
       self.END_LINE_TOKEN = "EL"
       self.START_TOKEN = "S"
       self.END_TOKEN = "E"
        
    def removeCharacters(self, characters):
        
        to_be_removed = "["
        for char in characters:
            to_be_removed+=char
        to_be_removed+="]"
        
        self.text = re.sub(to_be_removed, "", self.text)
        
        self.text = self.text.replace("‘","'")
        self.text = self.text.replace("’","'")
        self.text = self.text.replace("—","")
        self.text = self.text.replace("\"","")
        self.text = self.text.replace("'","' ")
        self.text = self.text.replace("'  ","' ")
        
    def lowerCase(self):
        
        self.text = self.text.lower()
        
    def splitLines(self):
        
        self.lines = self.text.splitlines()
        
    def removeSingleLines(self):
        
        new_lines = []
        for i in range(len(self.lines)-2):
            if self.lines[i]!="" and self.lines[i+1]=="" and self.lines[i+2]=="":
                pass
            else:
                new_lines.append(self.lines[i])
                
        self.lines = new_lines
        
    def removeFirstLines(self, n):
        
        for i in range(n):
            self.lines.pop(0)
        
    def addSpecialTokens(self):
        
        new_lines = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            rest = i%3
            if rest==0:
                new_lines.append(self.START_LINE_TOKEN + "A " + line + " " + self.END_LINE_TOKEN + "A")
            elif rest==1:
                new_lines.append(self.START_LINE_TOKEN + "B " + line + " " + self.END_LINE_TOKEN + "B")
            else:
                new_lines.append(self.START_LINE_TOKEN + "C " + line + " " + self.END_LINE_TOKEN + "C")
                
        self.lines = new_lines
            
        
    def addStartEndTokens(self):
        
        self.lines = [self.START_TOKEN+ line + self.END_TOKEN for line in self.lines]
        
    def removeEmptyLines(self):
        
        self.lines = [line for line in self.lines if line!="" and "•" not in line]
        
    def removePaddingStart(self):
        
        new_lines = []
        
        for line in self.lines:
            padd_number = 0
            while line[padd_number]==" ":
                padd_number+=1
                
            new_line=line[padd_number:]
            new_lines.append(new_line)
            
        self.lines = new_lines
        
    def removePaddingEnd(self):
        
        new_lines = []
        
        for line in self.lines:
            padd_number = 0
            while line[-1-padd_number]==" ":
                padd_number+=1
                
            new_line=line[:-padd_number]
            new_lines.append(new_line)
            
        self.lines = new_lines
            
    def saveText(self, title):
        
        f = open(title+".txt", "w",encoding="utf8")
        f.write(self.text)
        f.close()
        
    def saveLines(self, title):
        
        if not self.lines:
            return
        
        f = open(title+".txt", "a",encoding="utf8")
        
        for line in self.lines:
            f.write(line+"\n")
        
        f.close()