import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
movies_df = pd.read_csv('C:/Users/hp/Anaconda3/ml-10M100K/movies.dat', sep='::', header=None, engine='python')
print(movies_df.head())

ratings_df = pd.read_csv('C:/Users/hp/Anaconda3/ml-10M100K/ratings.dat', sep='::', header=None, engine='python')
print(ratings_df.head())
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
print(movies_df.head())
print(ratings_df.head())
print('The Number of Movies in Dataset', len(movies_df))
movies_df['List Index'] = movies_df.index
print(movies_df.head())
merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)
print(merged_df.head())
user_Group = merged_df.groupby('UserID')
print(user_Group.head())

amountOfUsedUsers = 1000
trX = []
for userID, curUser in user_Group:
    temp = [0]*len(movies_df)
    for num,movie in curUser.iterrows():
        #print(movie)
        temp[movie['List Index']] = movie['Rating']/5.0
    trX.append(temp)
    if(amountOfUsedUsers==0):
        break
    amountOfUsedUsers -= 1
print(trX)

hiddenUnits = 50
visibleUnits = len(movies_df)
vb = tf.placeholder(tf.float32, [visibleUnits])
hb = tf.placeholder(tf.float32, [hiddenUnits])
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)
err = v0 - v1
err_sum = tf.reduce_mean(err*err)
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
cur_vb = np.zeros([visibleUnits], np.float32)
cur_hb = np.zeros([hiddenUnits], np.float32)
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
prv_vb = np.zeros([visibleUnits], np.float32)
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()


inputUser = [trX[50]]
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
scored_movies_df_50 = movies_df
scored_movies_df_50["Recommendation Score"] = rec[0]
print(scored_movies_df_50.sort_values(["Recommendation Score"], ascending=False).head(20))
print(merged_df.iloc[50])
movies_df_50 = merged_df[merged_df['UserID'] == 150]
print(movies_df_50.head())
merged_df_50 = scored_movies_df_50.merge(movies_df_50, on='MovieID', how='outer')
merged_df_50 = merged_df_50.drop('List Index_y', axis=1).drop('UserID', axis=1)
print(merged_df_50.sort_values(['Recommendation Score'], ascending=False).head(20))
