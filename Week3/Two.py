import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset x
data = sns.load_dataset("iris")

# draw lineplot
sns.lineplot(x="sepal_length", y="sepal_width", data=data)
plt.title('Title using Matplotlib Function')
plt.xlim(5)
sns.set_style("dark")
plt.show()

titanic = sns.load_dataset("titanic")
sns.barplot(x="class", y="fare", hue="sex", palette="muted", data=titanic)
plt.xlabel("Class")
plt.ylabel("Fare")
plt.title("Average Fare by Class and Gender on the Titanic")
plt.show()


flights = sns.load_dataset('flights')
flights = flights.pivot('month', 'year', 'passengers')
sns.heatmap(flights, cmap='Blues', annot=True, fmt='d')
plt.title('Passengers per month')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()


tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", hue="time", data=tips, palette="Set3", linewidth=1.5, fliersize=4)
plt.title("Box Plot of Total Bill by Day and Meal Time")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill ($)")
plt.show()