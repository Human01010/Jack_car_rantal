import numpy as np

n = 10 # number of locations
days = 20

rent_means=np.random.randint(1, 6, n)
return_means=np.random.randint(1, 6, n)
car_num=4*np.ones(n) # suppose each location has 4 cars
data=np.zeros((days,n),dtype='int')

for day in range(days):
    rents=np.random.poisson(rent_means)
    returns=np.random.poisson(return_means)
    for i in range(n):
        real_rent=min(car_num[i],rents[i])
        real_return=returns[i]

        car_num[i]-=real_rent
        car_num[i]+=real_return
        data[day,i]=car_num[i]

print(("rent possion mean of each location :{}").format(rent_means))
print(("return possion mean of each location :{}").format(return_means))
print(("car number of each location at the end of each day :\n{}").format(data))