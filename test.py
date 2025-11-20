import numpy as np 
import matplotlib.pyplot as plt 

weather_data = np.load('data/parsed_data/weather_data.npy')
output_data = np.load('data/parsed_data/output_data.npy')



print(weather_data.shape, output_data.shape)


L = 30
weather_wins = np.lib.stride_tricks.sliding_window_view(weather_data, L, axis=0)
output_wins = np.lib.stride_tricks.sliding_window_view(output_data, L, axis=1)


print(weather_wins.shape) 
print(output_wins.shape)
quit()

plt.subplot(2,1,1)
plt.plot(weather_data[:,2])

plt.subplot(2,1,2)
plt.plot(output_data.mean(axis=(0,2)))
plt.show()
