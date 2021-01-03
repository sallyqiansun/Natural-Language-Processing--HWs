"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        ls=list()
        for key in self.rhs_to_rules.keys():
            if len(key)==2:
                upper1=True
                upper2=True
                for i in range(len(key[0])):
                    if key[0][i].isupper()==False and key[0][i].islower()==False:
                        continue
                    elif key[0][i].isupper()==True:
                        continue
                    else:
                        upper1=False
                if upper1==True:
                    for i in range(len(key[1])):
                        if key[1][i].isupper()==False and key[1][i].islower()==False:
                            continue
                        elif key[1][i].isupper()==True:
                            continue
                        else:
                            upper2=False
                    if upper2==True:
                        ls.append(key[0])
                        ls.append(key[1])


        for key in self.lhs_to_rules.keys():
            rules=self.lhs_to_rules[key]
            prob=list()
            for rule in rules:
                if len(rule[1])==1:
                    if rule[1][0] in ls:
                        return False
                    
                elif len(rule[1])==2:
                    if (rule[1][0] not in ls) or (rule[1][1] not in ls):
                        return False
                elif len(rule[1])>2 or len(rule[1])<1:
                        return False                                   
                prob.append(rule[2])
            if math.isclose(sum(prob),1,abs_tol=1e-05)!=True:
                return False                               
            else:
                continue
        return True


if __name__ == "__main__":
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        if grammar.verify_grammar()==True:
            print('Valid')
        else:
            print('Invalid')
        
