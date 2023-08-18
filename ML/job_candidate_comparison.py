import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import contractions
from googletrans import Translator
import os.path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import euclidean
import seaborn as sns
from simple_salesforce import Salesforce

class SalesForce(object):
	def __init__(self):
		self.sf = Salesforce(
			username='SALESFORCE_API_USER', 
			password = 'SALESFORCE_API_PASSWORD',
			security_token='SALESFORCE_API_TOKEN',
        )
		field_names = self.sf.UserInstall__c.describe()
		print([field['name'] for field in field_names['fields']])
	
	def query(self):
		return self.sf.query_all("""
			Select
			from UserInstall__c
			where
    	""")

class Node(object):
	def __init__(self, key, co):
		self.key = key
		self.left = None
		self.right = None
		self.height = 1
		self.company = co

class AVLTree(object):
	def insert(self, root, key, co):
		if not root:
			return Node(key, co)
		elif key < root.key:
			root.left = self.insert(root.left, key, co)
		else:
			root.right = self.insert(root.right, key, co)
			root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
		balance_factor = self.get_bal(root)
		if (balance_factor > 1):
			if key < root.left.key:
				return self.right_rotate(root)
			else:
				root.left = self.left_rotate(root.left)
				return self.right_rotate(root)
		if balance_factor < -1:
			if key > root.right.key:
				return self.left_rotate(root)
			else:
				root.right = self.right_rotate(root.right)
				return self.left_rotate(root)
		return root

	def delete(self, root, key):
		if not root:
			return root
		elif key < root.key:
			root.left = self.delete(root.left, key)
		elif key > root.key:
			root.right = self.delete(root.right, key)
		else:
			if root.left is None:
				temp = root.right
				root = None
				return temp
			elif root.right is None:
				temp = root.left
				root = None
				return temp
			temp = self.get_min_value(root.right)
			root.key = temp.key
			root.right = self.delete(root.right, temp.key)
		if root is None:
			return root
		root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
		balanceFactor = self.get_bal(root)
		if balanceFactor > 1:
			if self.get_bal(root.left) >= 0:
				return self.right_rotate(root)
			else:
				root.left = self.left_rotate(root.left)
				return self.right_rotate(root)
		if balanceFactor < -1:
			if self.get_bal(root.right) <= 0:
				return self.left_rotate(root)
			else:
				root.right = self.right_rotate(root.right)
				return self.left_rotate(root)
		return root

	def left_rotate(self, z):
		y = z.right
		if y is None:
			return z
		temp = y.left
		y.left = z
		z.right = temp
		z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
		y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
		return y

	def right_rotate(self, z):
		y = z.left
		if y is None:
			return z
		temp = y.right
		y.right = z
		z.left = temp
		z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
		y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
		return y

	def get_height(self, root):
		if not root:
			return 0
		return root.height

	def get_bal(self, root):
		if not root:
			return 0
		return self.get_height(root.left) - self.get_height(root.right)

	def get_min_value(self, root):
		if root is None or root.left is None:
			return root
		return self.get_min_value(root.left)

	def vals(self, root, pairings):
		if not root:
			return
		self.vals(root.right, pairings)
		pairings[root.company] = root.key
		self.vals(root.left, pairings)

class Transformers(object):
	def __init__(self, model):
		self.model = model

	def text_cleaning(self, input_string):
		input_string = input_string.lower()
		contractions.fix(input_string)
		input_string = re.sub('https?://\S+|www\.\S+', '', input_string)
		input_string = re.sub('\[.*?\]', '', input_string)
		input_string = re.sub("\\W"," ", input_string)
		input_string = re.sub('<.*?>+', '', input_string)
		input_string = re.sub('\n', '', input_string)
		input_string = re.sub('\w*\d\w*', '', input_string)
		input_string = re.sub('\s+', ' ', input_string)
		input_string = re.sub(r'[^A-Za-z0-9]+', ' ', input_string)
		return input_string

	def stop_word_lemmatize(self, input_string):
		stop = stopwords.words('english')
		input_string = " ".join([word for word in input_string.split() if word not in (stop)])
		words = word_tokenize(input_string)
		wnl = WordNetLemmatizer()
		out_text = []
		for w in words:
			out_text.append(wnl.lemmatize(w, pos="v"))
		return " ".join(out_text)

	def translate_text(self, text, target_language='en'):
		n, left, right = len(text), 0, 0
		out_text = set()
		translator = Translator()
		while left < n:
			# translates in batches of 100
			right += 100
			for word in text.split()[left:right]:
				if not word.isascii():
					word = translator.translate(word, src='ja', dest=target_language)
				out_text.add(word)
			left = right
		return  " ".join(out_text)

class JobTransformer(Transformers):
	def __init__(self, jdf, model=SentenceTransformer('all-mpnet-base-v2')):
		Transformers.__init__(self, model)
		self.jdf = jdf

	def run(self):
		# dict: id -> list of encodings + summary
		id_emb_dict = {}
		if os.path.isfile('job_tensors.pt'):
			id_emb_dict = torch.load('job_tensors.pt')
		for i in range(self.jdf.shape[0]):
			job_id = self.jdf.at[i, 'Id']
			if job_id in id_emb_dict:
				continue
			s = self.jdf.loc[i].str.cat(sep=" ")
			s = s.split(" ", 1)[1:]
			s = " ".join(s)
			s = re.sub("(<).*?(>)", "\g<1>\g<2>", s)
			s = self.text_cleaning(s)
			s = self.translate_text(s)
			s = self.stop_word_lemmatize(s)
			id_emb_dict.setdefault(job_id, [])
			id_emb_dict[job_id].extend([self.model.encode(s, convert_to_tensor=True), s])
		if id_emb_dict:
			torch.save(id_emb_dict, "job_tensors.pt")

class CandidateTransformer(Transformers):
	def __init__(self, cdf, model=SentenceTransformer('all-mpnet-base-v2')):
		Transformers.__init__(self, model)
		self.cdf = cdf

	def run(self):
		id_emb_dict = {}
		if os.path.isfile('candidate_tensors.pt'):
			id_emb_dict = torch.load('candidate_tensors.pt')
		for i in range(len(self.cdf)):
			candidate_id = self.cdf.at[i, "Id"]
			if candidate_id in id_emb_dict:
				continue
			s = self.cdf.loc[i].str.cat(sep=" ")
			s = s.split(" ", 3)[3:]
			s = " ".join(s)
			s = self.text_cleaning(s)
			s = self.translate_text(s)
			s = self.stop_word_lemmatize(s)
			id_emb_dict.setdefault(candidate_id, [])
			id_emb_dict[candidate_id].extend([self.model.encode(s, convert_to_tensor=True), s])
		if id_emb_dict:
			torch.save(id_emb_dict, "candidate_tensors.pt")

cos = torch.nn.CosineSimilarity(dim=0)
candidates = torch.load('candidate_tensors.pt')
jobs = torch.load('job_tensors.pt')
MAX_SUGGESTIONS = 5

def one_candidate(candidate_id):
	cols = {'JobId':[], 'Probability':[]}
	write_matches_df = pd.DataFrame(cols)
	match_tree = AVLTree()
	root = None
	count = 1
	for job_id, job_tensor in jobs.items():
		match_prob = cos(candidates[candidate_id][0], job_tensor[0]).item()
		root = match_tree.insert(root, match_prob, job_id)
		if count > MAX_SUGGESTIONS:
			root = match_tree.delete(root, match_tree.get_min_value(root).key)
		count += 1
	pairs = {}
	match_tree.vals(root, pairs)
	pair_count = 0
	for jid, prob in pairs.items():
		write_matches_df.loc[pair_count] = jid, prob
		pair_count += 1
	write_matches_df.to_csv(candidate_id + ".csv", index=False)

def specific_jobs(candidate_id, job_ids):
	cols = {'JobId':[], 'Probability':[]}
	write_matches_df = pd.DataFrame(cols)
	match_tree = AVLTree()
	root = None
	for job_id in job_ids:
		match_prob = cos(candidates[candidate_id][0], jobs[job_id][0]).item()
		root = match_tree.insert(root, match_prob, job_id)
	pairs = {}
	match_tree.vals(root, pairs)
	pair_count = 0
	for jid, prob in pairs.items():
		write_matches_df.loc[pair_count] = jid, prob
		pair_count += 1
	write_matches_df.to_csv(candidate_id + "_specifics.csv", index=False)

def all_matches():
	cols = {'CandidateId':[], 'JobId':[], 'Probability':[], 'CandidateUrl':[], 'JobUrl':[]}
	write_matches_df = pd.DataFrame(cols)	
	pair_count = 0
	for candidate_id, clist in candidates.items():
		match_tree = AVLTree()
		root = None
		count = 1
		for job_id, job_tensor in jobs.items():
			match_prob = cos(clist[0], job_tensor[0]).item()
			root = match_tree.insert(root, match_prob, job_id)
			if count > MAX_SUGGESTIONS:
				root = match_tree.delete(root, match_tree.get_min_value(root).key)
			count += 1
		pairs = {}
		match_tree.vals(root, pairs)
		for job_id, prob in pairs.items():
			prob = str(prob)[:4]
			candidate_url = "https://best-recruiters.lightning.force.com/lightning/r/Contact/" + str(candidate_id) + "/view"
			job_url = "https://best-recruiters.lightning.force.com/lightning/r/TR1__Job__c/" + str(job_id) + "/view"
			write_matches_df.loc[pair_count] = candidate_id, job_id, prob, candidate_url, job_url
			pair_count += 1
	write_matches_df.to_csv("all_matches" + ".csv", index=False)

def fine_tune(model=SentenceTransformer('all-mpnet-base-v2')):
	train_matches = [
		InputExample(texts=[candidates['0035h00001968tWAAQ'][1], jobs['a0s5h000002AHa8AAG'][1]], label=0.9),
		InputExample(texts=[candidates['0035h00001968tWAAQ'][1], jobs['a0s5h0000022lLiAAI'][1]], label=0.1),
	]
	train_dataloader = DataLoader(train_matches, shuffle=True)
	train_loss = losses.CosineSimilarityLoss(model)
	model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, output_path="C:\\Users\\ivan_apexes\\Desktop\\clonebot\\new_model")
	return model

if __name__ == "__main__":
	# model = fine_tune()
	# job_input_path = ""
	# candidate_input_path = ""

	job_input_path = "C:\\Users\\ivan_apexes\\Downloads\\100 Contacts.xlsx - 1000 jobs.csv"
	candidate_input_path = "C:\\Users\\ivan_apexes\\Downloads\\100 Contacts.xlsx - 1000 candidates.csv"
	jdf = pd.read_csv(job_input_path)
	while (jdf.columns[0] != "Id"):
		jdf = jdf.drop(jdf.columns[0], axis=1)
	jdf.to_csv(job_input_path, index=False)
	cdf = pd.read_csv(candidate_input_path)
	while (cdf.columns[0] != "Id"):
		cdf = cdf.drop(cdf.columns[0], axis=1)
	cdf.to_csv(candidate_input_path, index=False)
	JobTransformer(jdf).run()
	CandidateTransformer(cdf).run()

	# all_matches()
	# specific_jobs('0035h00001968tWAAQ',['a0s5h0000022lK7AAI','a0s5h0000022lRtAAI','a0s5h0000022lLiAAI'])
	# one_candidate('0035h00001968tWAAQ')
	
	# k nearest jobs to candidate
	data = torch.load('job_tensors.pt')
	X, ids = [], []
	for i, j in data.items():
		ids.append(i), X.append(j[0].numpy())

	NEAREST_NEIGHBORS = MAX_SUGGESTIONS
	data = torch.load('candidate_tensors.pt')
	candidate = data['0035h00001968tWAAQ'][0]
	kmeans = KMeans(init = 'k-means++', n_init='auto', n_clusters=10, random_state=34)
	kmeans.fit(X)
	distances = [euclidean(candidate, centroid) for centroid in kmeans.cluster_centers_]
	closest_cluster = np.argmin(distances)
	cluster_points = np.where(kmeans.labels_ == closest_cluster)[0]
	distances_within_cluster = [euclidean(candidate, kmeans.cluster_centers_[closest_cluster]) for i in cluster_points]
	closest_points_indices = np.argsort(distances_within_cluster)[:NEAREST_NEIGHBORS]
	print(ids[i] for i in closest_points_indices)
	
	# dim reduction for visualization
	X = normalize(X)
	X = StandardScaler().fit_transform(X)
	pca = PCA(n_components=2)
	X = pca.fit_transform(X)
	print(pca.explained_variance_ratio_)
	
	# scatterplot
	k = 10
	km = KMeans(init = 'k-means++', n_init='auto', n_clusters=k, random_state=34)
	km.fit(X)
	label = km.labels_
	k = 17
	ul = np.unique(label)
	for i in ul:
		plt.scatter(X[label == i, 0], X[label == i, 1], label = i)
	plt.show()

	# elbow method
	sum_square_error = []
	for ki in range(2, k):
		kmi = KMeans(init='k-means++', n_init='auto', n_clusters=ki, random_state=34)
		kmi.fit(X)
		lbl = kmi.fit_predict(X)
		sum_square_error.append(kmi.inertia_)
	sns.set_style("whitegrid")
	graph = sns.lineplot(x=range(2, k), y=sum_square_error, marker='o')
	graph.set(xlabel="Cluster Size", ylabel="SSE")
	plt.show()

	# silhouette analysis
	cluster_labels = np.unique(label)
	num_clusters = cluster_labels.shape[0]
	silhouette_vals = silhouette_samples(X, label)
	y_ax_lower, y_ax_upper = 0, 0
	y_ticks = []

	for idx, cls in enumerate(cluster_labels):
		cls_silhouette_vals = silhouette_vals[label==cls]
		cls_silhouette_vals.sort()
		y_ax_upper += len(cls_silhouette_vals)
		cmap = cm.get_cmap("Spectral")
		rgba = list(cmap(idx/num_clusters))
		rgba[-1] = 0.7
		plt.barh(
			y=range(y_ax_lower, y_ax_upper), 
			width=cls_silhouette_vals,
			height=1.0,
			edgecolor='none',
			color=rgba)
		y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
		y_ax_lower += len(cls_silhouette_vals)

	silhouette_avg = np.mean(silhouette_vals)
	plt.axvline(silhouette_avg, color='orangered', linestyle='--')
	plt.xlabel('Silhouette Coefficient')
	plt.ylabel('Cluster Size')
	plt.yticks(y_ticks, cluster_labels + 1)
	plt.show()
