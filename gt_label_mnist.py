import os
import pandas as pd 


def area(df,index, label):
    x = df["area"].iloc[index]
    if label == 1: #very_small_area(x) = x.area<30, small_area(x) = 30<= x.area<50, large_area(x) = 50<=x.area<100, very_large_area(x) = x.area>=100
        if x<30:
            return "very small area"
        elif x<50:
            return "small area"
        elif x<100:
            return "large area"
        else:
            return "very large area"
    else: #very_small_area(x) = x.area<30, small_area(x) = 30<= x.area<100, large_area(x) = 100<=x.area<300, very_large_area(x) = x.area>=300
        if x<30:
            return "very small area"
        elif x<100:
            return "small area"
        elif x<300:
            return "large area"
        else:
            return "very large area"

def length(df,index):
    x = df["length"].iloc[index] #extreme_short(x) = x.length<10 , short(x) = 10<=x.length<35 , long(x) = 35<=x.length<50 , extreme_long(x) = x.length>50
    if x <10:
        return "very short length"
    elif x<35:
        return "short length"
    elif x<50:
        return "long length"
    else:
        return "very long length"

def thickness(df,index):
    x = df["thickness"].iloc[index] #extreme_thin(x) = x.thickness<=1.5, thin(x) = 1.5<x.thickness<3.5, slightly_thick(x) = 3.5<=x.thickness<4, thick(x) = 4<=x.thickness<7, extreme_thick(x) = x.thickness>=7
    if x<=1.5:
        return "very thin"
    elif x<3.5:
        return "thin"
    elif x<4:
        return "slightly thick"
    elif x<7:
        return "thick"
    else:
        return "very thick"

def slant(df,index):
    x = df["slant"].iloc[index] #upright(x) = |x.slant|<=0.1, left(x) = -0.4<x.slant<-0.1, right(x) = 0.1<x.slant<0.4, extreme_left(x) = x.slant<=-0.4, extreme_right(x) = x.slant>=0.4
    if x>=-0.1 and x<=0.1:
        return "upright"
    elif x>-0.4 and x<-0.1:
        return "left leaning"
    elif x>0.1 and x<0.4:
        return "right leaning"
    elif x<=-0.4:
        return "very left leaning"
    elif x>=0.4:
        return "very right leaning"

def width(df,index, label):
    x = df["width"].iloc[index] 
    if label ==1: #extreme_narrow(x) = x.width<3.5, narrow(x) = 3.5<=x.width<5, wide(x) = 5<=x.width<7.5, extreme_wide(x) = x.width>=7.5
        if x<3.5:
            return "very narrow"
        elif x<5:
            return "narrow"
        elif x<7.5:
            return "wide"
        else:
            return "very wide"
    
    else:#extreme_narrow(x) = x.width<9, narrow(x) = 9<=x.width<12, wide(x) = 12<=x.width<18, extreme_wide(x) = x.width>=18
        if x<9:
            return "very narrow"
        elif x<12:
            return "narrow"
        elif x<18:
            return "wide"
        else:
            return "very wide"

def height(df,index):
    x = df["height"].iloc[index] #x.height<14, low(x) = 14<=x.height<17, high(x) = 17<= x.height(x)<20, extreme_high(x) = x.height>=20     
    if x<14:
        return "very low height"
    elif x<17:
        return "low height"
    elif x<20:
        return "high height"
    else:
        return "very high height"


    





