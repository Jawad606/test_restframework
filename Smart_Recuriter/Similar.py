import textdistance as td
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
import pandas as pd
# def match(resume, job_des):
#     j = jaccard_score(resume, job_des)
#     # s = td.sorensen_dice.similarity(resume, job_des)
#     c =cosine_similarity(resume,job)
#     # o = td.overlap.normalized_similarity(resume, job_des)
#     total = (j+c+o)/4
#     # total = (s+o)/2
#     return total

# 'sentence-transformers/all-MiniLM-L6-v2
def match(resume,job_des):
    print(resume,job_des)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    resume = model.encode(resume, batch_size = 10, show_progress_bar = True)
    job = model.encode(job_des, batch_size =2, show_progress_bar = True)
    similarities = cosine_similarity(resume,job)
    return similarities * 100


# job="""able advanced algorithm amount analysis analytics apply appropriate approval audio bachelor bayes begin build business candidate cassandra change client cnn common communication companys conduct crunching cut data databacke database datum decision define deliver deliverable design develop diagnostic discover distribution edge education enable energetic engineer even excellent experience experiment focus forests fulltime game gamechange generate ggplot good grasp great gurgaon haryana hbase help hide high hive identify implementation implementationfinetune improvement include independently information innovative insight integrate internal job keep language learn learning liaise limit look lstm machine make massaging matlab meaningful mining ml model naive nn nosql numerical numpy opencv operation opportunity performance pig plan prediction preferred present primary product proficiency programming python quality query quickly record regression remote require resourceful responsibilities rnn salary science sciencemachine scientist script scripting selfstarter skill skills smart solutions specific sql statistic statistical strategy structure supervision svm system tactical team technique technology temporarily testing text tool toolkit type understand unstructured use vast video vision visualisation warehousing weka well work year"""
 
# resume="""1sep20 1sep20last accuracy active ambedkar analyse analysismachine application area automation automobile back bangalore bangalorehyderabadpune bengaluru capture career cas cassandra cognitive company computer computers cto5 current data date datum deep degree design designation designingapplication developer development developmentdata developmentpythonmysqlkafkanosqlmongodbmachine developmentsoftware different divya dr education email engineer english exist experience facial file focus functional future geofence handicapped highest hindi hopes id improved industry institute intelligencepattern it itsoftwaresoftware json jul jump jun key known language languages last learn learning learningcomputer learningdeep learningnatural licence location log machine maintenance marital model modeling modified month months mysql name nlp nlpcomputer notice number ocr pattern period phone pipeline plate prakash predict predictive pref process processingartificial professional proficiency programming programmingcodingweb python rd read recognition recognitiondata replace rest resume role rule science sectionwork seek services several singhfeature singleunmarried skill skilled skills soap software solution speak status summary technology tool top total traffic ug unisys use verify version vision visionpythonmysqljson visionpythonmysqljsonapisoftware visionsoftware voilate voilation wipro work write year years"""
 

# result=match(job,resume)
 
 
 
 
 
 
 
 
 
 
 
 