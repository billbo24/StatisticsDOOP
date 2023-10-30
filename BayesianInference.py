#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:24:15 2023

@author: williamfloyd
Alright the statistics page for Bayesian inference has a great example that I want to 
replicate here.

The problem:
    You are an archaeologist trying to date a dig site between the 11th and 16th century.  
    You know that 1% of all pottery was glazed and 50% of all pottery was decorated in
    the 11th centure, and that 81% was glazed and 5% decorated in the 16th century.  These
    percentages change continuously through time, so we can map them as
    
    p(glazed | c) = (c-11)*((0.81-0.01)/(16-11))+0.01
    p(decorated | c) = 0.5 - (c-11)*((0.50-0.05)/(16-11))
    
    likewise, the probability of BOTH is just those two multiplied togther (naturally we're
    assuming independence between the two variables).  Similarly we can subtract those 
    from one to get the probabilities they don't exhibit those traits.  

    Wikipedia  use the following notation: G = glazed, G* = unglazed, D = decorated
    D* = undecorated.  When we find a shard, it can only be one of four possibilities                                                                        
    {GD,GD*,G*D,G*D*}.  

    Now the problem is to discover the century, and we will attempt to do this by finding the
    function f(c), the pdf for the century.  Letting e be the evidence we get (pot shard) and 
    and c be the century, we have the following from Bayes Law:
    
    P(C = c | E = e) = P(E = e | C = c)*P(C=c) / P(E=e)
                     = P(E = e | C = c) * f(c) / (int(11,16) P(E=e)*f(c)dc)

    This may look really intimidating, but it's really not that bad.  The denominator 
    is a basic expected value term, intgerating over all values of c, and P(C=c) is the
    function we are after.  Note that we can now iteratively change our guess for f(c)
    using this equation.  We begin with a simple estimate that f(c) = 0.2, namely
    that every value is equally likely.  Then as we draw a new pottery shard, e, we can
    compute the probability for a number of different c values to get our new f(c) function

    The ultimate idea here is that as we pull more and more say glazed shards, the likelihood
    of the site being from the 11th century goes down a ton.  


    Update: Think I'm gonna modify the problem.  Either way though, gonna build some of the foundational
    functions                      
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sci


#Alright, I think I'm gonna try and make this at least some what general purpose
def initialize_stuff():
    left_endpoint = 0
    right_endpoint = 10
    num_steps = 100 #this will be how many places we evaluate the functions
    return np.linspace(left_endpoint,right_endpoint,num_steps,endpoint=True),left_endpoint,right_endpoint




def pdf(x): 
    #This will be our pdf.  Note that this will be updating 
    return 1


def glaze_pdf(c):
    #Glazed is more likely later on
    m = (0.19-0.01)/10
    return m*c + 0.01 #y = mx+b


def painted_pdf(c):
    return (0.04578949240898836)*(math.sin(c) + 2)


    
#This function will change depending on problem
#calling it c for now, hypothetically iterating over 10 centuries
def prob_evidence_given_c(evidence,c):
    num1 = glaze_pdf(c)
    num2 = painted_pdf(c)
    
    if evidence[0] == 'glazed':
        if evidence[1] == 'painted':
            return num1*num2
        else:
            return num1*(1-num2)
    else: #not glazed
        if evidence[1] == 'painted':
            return (1-num1)*num2
    return (1 - num1)*(1-num2)
    
    

def integrate_func(xs,ys):
    #xs and ys need same dimension
    #Just goin trapezoid.  Might wanna do parabolas if this stinks
    ans = 0
    for i in range(0,len(xs)-1):
        width = xs[i+1]-xs[i]
        area = width*(ys[i]+ys[i+1])/2
        ans += area
    return ans

'''
aa = np.linspace(0, 2,100)
bb = [i**2 for i in aa]
area = integrate_func(aa,bb)
print(area)
'''    
    
def iterate_evidence(evidence):
    
    #dumb bullshit because I'm a lazy coder
    f = lambda x: math.sin(x) + 2
    k = sci.integrate.quad(f,0,10) #bounds, need to make more general
    print(1/k[0])

    
    #get our initial points
    x_points,left_end,right_end = initialize_stuff()
    
    xs = [x for x in x_points]
    #This gives us the normalized constant so our pdf has an area of 1ß
    #This will more or less be our seed pdf
    constant = 1 / (right_end-left_end)
    g = lambda x: constant
    
    #this gives us our corresponding starter pdf for the c's, not evidence
    y_points = [g(x) for x in x_points]
    
    #now recall the idea is that as we iterate over evidence, each piece of evidence
    #will change the result.  In the above problem, we're looking for f(century|evidence)
    #this equals (f(evidence|century)*f(century))/p(evidence).  
    
    #We need a way to map a 
    #piece of evidence to an output probability/pdf (e.g. a glazed shard has a 3% chance of
    #coming from the 11th century, 10% chqnce of 12th etc)
    
    #For each piece of evidence we have to compute P(evidence), and in the above problem
    #This is taking the integral P(shard|century)*f(century)dcentury.  Note this is a constant
    #for each piece of evidence.  I think I can get away with riemann sums for the area
    
    #Then all we need to do is iterate over all of our x points, compute P(Evidence|x) 
    #and then replace our new point with an old one.  
    
    fig,ax = plt.subplots()
    counter = 1
    
    for e in evidence:
        #first step is compute probability of evidence by integrating over 
        #all x values.
        #Also it's really stupid to do it this way with the indexing and whatnot, but 
        #I can't think of a better way at the moment lol
        h = lambda c,e: prob_evidence_given_c(e,c)*y_points[xs.index(c)]
        
        temp = [h(x,e) for x in x_points]
        bottom_term = integrate_func(xs, temp)
        
        
        print(e,bottom_term)
        new_ys = []
        
        for c in xs:
            term1 = prob_evidence_given_c(e, c)
            term2 = y_points[xs.index(c)] #applying f(c)
            new_y = (term1*term2)/bottom_term #This is the Bayes step
            new_ys.append(new_y)
        label = f"data point {counter}"
        ax.plot(x_points,new_ys,label=label)
        counter += 1
        
        #This is how we iterate
        y_points = new_ys
    ax.legend()

a = [["glazed","painted"],
     ["glazed","not painted"],
     ["not glazed","not painted"],
     ["not glazed","not painted"],
     ["not glazed","not painted"]]


iterate_evidence(a)



