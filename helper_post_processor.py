import pandas as pd

pixelThresh = .01
sideThresh = 0.95

neighbWords = ["W1Ab","W2Ab","W3Ab","W4Ab","W5Ab",
               "d1Ab","d2Ab","d3Ab","d4Ab","d5Ab",
               "W1Be","W2Be","W3Be","W4Be","W5Be",
               "d1Be","d2Be","d3Be","d4Be","d5Be",
               "W1Lf","W2Lf","W3Lf","W4Lf","W5Lf",
               "d1Lf","d2Lf","d3Lf","d4Lf","d5Lf",
               "W1Rg","W2Rg","W3Rg","W4Rg","W5Rg",
               "d1Rg","d2Rg","d3Rg","d4Rg","d5Rg",
               "is1lfAzLn","is2lfAzLn","is3lfAzLn","is4lfAzLn","is5lfAzLn",
               "is1rgAzLn","is2rgAzLn","is3rgAzLn","is4rgAzLn","is5rgAzLn"]

def findLeftWords(row,df2):

    #Left words
    dffilt = df2[(df2["right"] < row["left"]) & 
                 (df2["bottom"] >= row["top"]) & 
                 (df2["top"] <= row["bottom"]) &
                 (df2["OriginalFile"] == row["OriginalFile"]) &
                 (df2["page_num"] == row["page_num"])]
    dffilt = dffilt.sort_values(["right"],ascending = False)
    dffilt = dffilt.head(5)
    if dffilt.shape[0] == 0:
        words = [""] * 5
        dist = [0] * 5
        azln = [0] * 5
        return words,dist,azln
    else:
        dffilt["dist"] = dffilt["left"] - row["left"]
        dffilt["isAzLn"] = (dffilt["line_num"] == row["line_num"])
        dffilt["isAzLn"] = dffilt["isAzLn"].astype(int)

        words = list(dffilt["text"])
        words.extend([""] * (5 - len(words)))
        dist = list(dffilt["dist"])
        dist.extend([0] * (5 - len(dist)))
        azln = list(dffilt["isAzLn"])
        azln.extend([0] * (5 - len(azln)))
        return words,dist,azln

def findRightWords(row,df2):
    #Right words
    dffilt = df2[(df2["left"] > row["right"]) & 
                 (df2["bottom"] >= row["top"]) &
                 (df2["top"] <= row["bottom"]) &
                 (df2["OriginalFile"] == row["OriginalFile"]) &
                 (df2["page_num"] == row["page_num"])]
    dffilt = dffilt.sort_values(["left"],ascending = True)
    dffilt = dffilt.head(5)
    if dffilt.shape[0] == 0:
        words = [""] * 5
        dist = [0] * 5
        azln = [0] * 5
        return words,dist,azln
    else:
        dffilt["dist"] = dffilt["left"] - row["left"]
        dffilt["isAzLn"] = (dffilt["line_num"] == row["line_num"])
        dffilt["isAzLn"] = dffilt["isAzLn"].astype(int)

        words = list(dffilt["text"])
        words.extend([""] * (5 - len(words)))
        dist = list(dffilt["dist"])
        dist.extend([0] * (5 - len(dist)))
        azln = list(dffilt["isAzLn"])
        azln.extend([0] * (5 - len(azln)))
        return words,dist,azln

def findAboveWords(row,df2):

    #Words Above
    dffilt = df2[(df2["bottom"] < row["top"]) &
                 ((row["top"] - df2["bottom"]) <= pixelThresh) &
                 (df2["right"] >= row["left"]) &
                 (df2["left"] <= row["right"]) &
                 (df2["OriginalFile"] == row["OriginalFile"]) &
                 (df2["page_num"] == row["page_num"])]
    dffilt = dffilt.sort_values(["bottom","left"],ascending = [False,True])
    dffilt = dffilt.head(5)
    if dffilt.shape[0] == 0:
        words = [""] * 5
        dist = [0] * 5
        return words,dist
    else:
        dffilt["dist"] = dffilt["top"] - row["top"]

        words = list(dffilt["text"])
        words.extend([""] * (5 - len(words)))
        dist = list(dffilt["dist"])
        dist.extend([0] * (5 - len(dist)))
        return words,dist

def findBelowWords(row,df2):

    #Words Below
    dffilt = df2[(df2["top"] > row["bottom"]) & 
                 ((df2["top"] - row["bottom"]) <= pixelThresh) &
                 (df2["right"] >= row["left"]) &
                 (df2["left"] <= row["right"]) &
                 (df2["OriginalFile"] == row["OriginalFile"]) &
                 (df2["page_num"] == row["page_num"])]

    dffilt = dffilt.sort_values(["top","left"],
                                ascending = [True,True])
    dffilt = dffilt.head(5)
    if dffilt.shape[0] == 0:
        words = [""] * 5
        dist = [0] * 5
        return words,dist
    else:
        dffilt["dist"] = dffilt["top"] - row["top"]

        words = list(dffilt["text"])
        words.extend([""] * (5 - len(words)))
        dist = list(dffilt["dist"])
        dist.extend([0] * (5 - len(dist)))
        return words,dist


def findCloseWords(row, df2):

    result = []

    #words above
    abwords,abdist = findAboveWords(row,df2)
    result.extend(abwords)
    result.extend(abdist)
    #words below
    bewords,bedist = findBelowWords(row,df2)
    result.extend(bewords)
    result.extend(bedist)
    #Left words
    lftwords,lftdist,lftazln = findLeftWords(row,df2)
    result.extend(lftwords)
    result.extend(lftdist)
    #Right words
    rgtwords,rgtdist,rgtazln = findRightWords(row,df2)
    result.extend(rgtwords)
    result.extend(rgtdist)

    result.extend(lftazln)
    result.extend(rgtazln)
    return pd.Series(result)


def wordBoundingText(DF):
    """
    """
    DF[neighbWords] = DF.apply(findCloseWords, args=(DF,),axis = 1)
    return DF