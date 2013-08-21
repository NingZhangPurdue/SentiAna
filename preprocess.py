"""
Preprocess the comments data.
1, build feature index.
2, generate feature/label matrix.
"""

import sys
import codecs

featureTerms = set()
lineCount = 0
wordCount = 0
for file in sys.argv[1:3]:
    with codecs.open(file, 'r', 'utf-8') as input:
        for line in input:
            lineCount += 1
            for term in line.split('\t')[1].split():
                featureTerms.add(term)
                wordCount += 1
featureTerms = list(featureTerms)
termIndex = dict()
output = codecs.open('termIndex', 'w', 'utf-8')
for term in featureTerms:
    termIndex[term] = featureTerms.index(term)
    output.write(term + '\t' + str(featureTerms.index(term)) + '\n')
output.close()
termIndex = dict((term, featureTerms.index(term)) for term in featureTerms)
M = len(featureTerms)
print '%d unique terms.' % M
print '%d words each comment has in average.' % (wordCount / lineCount)
print '%d comments.' % lineCount

QQlabel = {'1': '1', '2': '2', '3': '3', '7': '4', '4': '5', '5': '6', '6': '7', '8': '8'}
Sinalabel = {'7': '1', '0': '2', '1': '3', '6': '4', '3': '5', '4': '6', '5': '7', '2': '8'}
for file in sys.argv[1:3]:
    print file
    output = open('feaMat_' + file, 'w')
    with codecs.open(file, 'r', 'utf-8') as input:
        for line in input:
            featureVec = [0] * M
            segs = line.split('\t')
            for term in segs[1].split():
                featureVec[termIndex[term]] = 1
            #print segs[0]
            if 'QQ' in file and segs[0] in QQlabel:
                #print segs[0], QQlabel[segs[0]]
                output.write(segs[2] + '\t' + QQlabel[segs[0]] + '\t' +
                             '\t'.join(str(tf) for tf in featureVec) + '\n')
            elif 'Sina' in file and segs[0] in Sinalabel:
                output.write(segs[2] + '\t' + Sinalabel[segs[0]] + '\t' +
                             '\t'.join(str(tf) for tf in featureVec) + '\n')
    output.close()
