from math import sqrt

 data = {
 			"Manish": {
 						"Interstellar": 4,
 						"The Dark Knight": 5,
 						"Wanted": 3,
 						"Sucker Punch": 2,
 						"Inception": 5,
 						"The Conjuring": 3,
 						"21 Jump Street": 4,
 						"The Prestige": 5
  					},
  			"Madhu": {
  						"Interstellar": 5,
 						"The Dark Knight": 5,
 						"Wanted": 1,
 						"Devil": 3,
 						"The Conjuring": 1,
 						"21 Jump Street": 4,
 						"Men in Black": 2

  					},
  			"Mansukh": {
  						"Hot Tub Time Machine": 1,
  						"Inception": 5,
  						"Revenant": 3,
  						"Avengers 1": 4,
  						"Iron Man 2": 3,
  						"Batman v Superman": 5,
  						"Wanted": 4,

  					},
  			"Imran": {
  						"Inception": 5,

  					},
  			"Kumar": {
  						"Hot Tub Time Machine": 1,
  						"Avengers 1": 4,
  						"Avengers 2": 3,
  						"The Departed": 5,
  						"Interstellar": 4,
  						"Fight Club": 5,
  						"Vampires Suck": 1,
  						"Twilight": 1
  					},
  			"Tori": {
  						"Notebook": 5,
  						"The Terminal": 4,
  						"Twilight": 5,
  						"Inception": 2,
  						"The Dark Knight": 1,
  						"Hot Tub Time Machine": 2,
  						"The Vow": 4
  					},
  			"Jatin": {
 						"Inception":5,
 						"The Conjuring":4
  					},
  			"Latha": {
  						"Twilight": 1
  					}
  		}

 itemNames = [
 				"Interstellar",
 				"The Dark Knight",
 				"Wanted",
 				"Sucker Punch",
 				"Inception",
 				"The Conjuring",
 				"21 Jump Street",
 				"The Prestige",
 				"Devil",
 				"Men in Black",
 				"Hot Tub Time Machine",
 				"Revenant",
  				"Avengers 1",
  				"Iron Man 2",
  				"Batman v Superman",
  				"Avengers 2",
  				"The Departed",
  				"Fight Club",
  				"Vampires Suck",
  				"Twilight",
  				"Notebook",
  				"The Terminal",
  				"The Vow",
  				"Focus"
 			]

 MAXrating = 5
 MINrating = 1

 def compute_similarity(item1,item2,userRatings):
 	averages = {}
 	for (key,ratings) in userRatings.items():
 		averages[key] = (float(sum(ratings.values()))/len(ratings.values()))

 	num = 0
 	dem1 = 0
 	dem2 = 0

 	for (user,ratings) in userRatings.items():
 		if item1 in ratings and item2 in ratings:
 			avg = averages[user]
 			num += (ratings[item1] - avg) * (ratings[item2] - avg)
 			dem1 += (ratings[item1] - avg) ** 2
 			dem2 += (ratings[item2] - avg) ** 2
 	if dem1*dem2 == 0:
 		return 0
 	return num / (sqrt(dem1 * dem2))

 def build_similarity_matrix(userRatings):
 	similarity_matrix = {}

 	for i in range(0,len(itemNames)):
 		band = {}
 		for j in range(0,len(itemNames)):
 			if itemNames[i] != itemNames[j]:
 				band[itemNames[j]] = compute_similarity(itemNames[i],itemNames[j],data)
 		similarity_matrix[itemNames[i]] = band
 	return similarity_matrix

 def normalize(rating):
 	num = 2 * (rating - MINrating) - (MAXrating - MINrating)
 	den = (MAXrating - MINrating)
 	return num / den

 def denormalize(rating):
 	return (((rating + 1) * (MAXrating - MINrating))/2 ) + MINrating

 def prediction(username,item):
 	num = 0
 	den = 0
 	for band,rating in data[username].items():
 		num += sm[item][band] * normalize(rating)
 		den += abs(sm[item][band])

 	if den == 0:
 		return 0
 	return denormalize(num/den)

 def recommendation(username,userRatings):
 	recommend = []
 	for item in itemNames:
 		if item not in userRatings[username].keys():
 			if prediction(username,item) >= 3.5:
 				recommend.append(item)
 	return recommend

 sm = build_similarity_matrix(data)
 # for k,i in sm.items():
 # 	print(k , i)
 print("Recommendation for Jatin: ")
 print(recommendation("Jatin",data))

 print("Recommendation for Latha: ")
 print(recommendation("Latha",data))
