# Importing Libraries and Modules
import torch #Library untuk komputasi tensor dan jaringan neural
import random #Library untuk generasi angka acak.
import numpy as np #Library untuk operasi pada array.
from collections import deque #Double-ended queue untuk menyimpan memori permainan.

#Modul khusus yang berisi implementasi permainan Snake, model jaringan neural, dan fungsi plotting.
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

#Constants
MAX_MEMORY = 100_000 #Batas maksimum memori yang digunakan untuk menyimpan pengalaman permainan.

BA  TCH_SIZE = 1000 #Batas maksimum memori yang digunakan

LR = 0.001 #Learning rate untuk pelatihan model.

#Agent Class
class Agent:
    def __init__(self):
        # Inisialisasi agen dengan parameter awal
        self.n_games = 0
        self.epsilon = 0  # Parameter untuk mengontrol keacakan dalam aksi
        self.gamma = 0.9  # Discount rate untuk Q-learning
        self.memory = deque(maxlen=MAX_MEMORY)  # Memori untuk menyimpan pengalaman permainan
        self.model = Linear_QNet(11, 256, 3)  # Model jaringan neural dengan ukuran input 11, ukuran hidden 256, dan ukuran output 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer Q-learning dengan model, learning rate, dan discount rate

    def get_state(self, game):
        # Mendapatkan keadaan saat ini dari permainan
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Mendapatkan arah saat ini dari ular
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # kombinasi dari informasi bahaya dan arah, serta lokasi makanan relatif terhadap kepala ular
        state = [
            # Bahaya lurus
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Bahaya kanan
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Bahaya kiri
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Arah gerak
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Lokasi makanan
            game.food.x < game.head.x,  # makanan di kiri
            game.food.x > game.head.x,  # makanan di kanan
            game.food.y < game.head.y,  # makanan di atas
            game.food.y > game.head.y  # makanan di bawah
        ]

        return np.array(state, dtype=int)

 def remember(self, state, action, reward, next_state, done):
        # Menyimpan pengalaman ke dalam memori
        self.memory.append((state, action, reward, next_state, done))  # Secara otomatis menghapus memori tertua jika batas maksimum tercapai

    def train_long_memory(self):
        # Melatih model menggunakan batch besar dari memori
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Mengambil sampel acak dari memori
        else:
            mini_sample = self.memory  # Jika memori kurang dari ukuran batch, gunakan semua memori

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Melatih model menggunakan satu langkah dari permainan
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Menentukan aksi yang akan diambil berdasarkan keadaan saat ini
        self.epsilon = 80 - self.n_games  # Mengurangi keacakan seiring waktu
        final_move = [0, 0, 0]  # Inisialisasi gerakan sebagai tidak ada aksi

        if random.randint(0, 200) < self.epsilon:
            # Mengambil aksi acak
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Memprediksi aksi terbaik berdasarkan keadaan saat ini
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    # Loop utama untuk pelatihan
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Mendapatkan keadaan saat ini
        state_old = agent.get_state(game)
        
        # Mendapatkan aksi yang akan diambil
        final_move = agent.get_action(state_old)
        
        # Melakukan aksi dan mendapatkan reward, done flag, dan skor
        reward, done, score = game.play_step(final_move)
        
        # Mendapatkan keadaan baru
        state_new = agent.get_state(game)
        
        # Melatih model dengan memori pendek (satu langkah)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Mengingat pengalaman permainan
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Jika permainan selesai, reset permainan
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()  # Melatih model dengan memori panjang (batch dari pengalaman)

            if score > record:
                record = score
                agent.model.save()  # Menyimpan model jika mencapai rekor baru

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)  # Plot skor

if __name__ == '__main__':
    train()  # Menjalankan loop pelatihan