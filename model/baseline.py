import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, users_max_info, movies_max_info, embedding_num, output_dim):
        super(Baseline, self).__init__()

        self.embedding_dim = embedding_num
        # 构建用户特征的embedding
        self.encoder_gender = nn.Embedding(users_max_info[0], self.embedding_dim)
        self.encoder_age = nn.Embedding(users_max_info[1], self.embedding_dim)
        self.encoder_occupation = nn.Embedding(users_max_info[2], self.embedding_dim)
        # self.encoder_zipcode = nn.Embedding(users_max_info[3], self.embedding_dim)
        # 构建电影特征的embedding
        self.encode_title = nn.EmbeddingBag(movies_max_info[0], self.embedding_dim, mode = 'mean')
        self.encode_genres = nn.EmbeddingBag(movies_max_info[1], self.embedding_dim, mode = 'mean')

        self.user_linear = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace= True),
        )
        self.movie_linear = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace= True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2*512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, output_dim),
        )

    def forward(self, user_feature, movie_feature):
        output_encoder_gender = torch.squeeze(self.encoder_gender(user_feature['gender']))
        output_encoder_age = torch.squeeze(self.encoder_age(user_feature['age']))
        output_encoder_occupation = torch.squeeze(self.encoder_occupation(user_feature['occupation']))

        output_encoder_title = self.encode_title(movie_feature['title'])
        output_encoder_genres = self.encode_genres(movie_feature['genres'])

        user_feature = torch.cat([output_encoder_gender, output_encoder_age, output_encoder_occupation], axis = -1)
        movie_feature = torch.cat([output_encoder_title, output_encoder_genres], axis = -1)


        user_feature = self.user_linear(user_feature)
        movie_feature = self.movie_linear(movie_feature)

        
        mix_feature = torch.cat([user_feature, movie_feature], axis = -1)
        mix_feature = self.fc1(mix_feature)
        mix_feature = self.fc2(mix_feature)
        output = self.fc3(mix_feature)
        return output