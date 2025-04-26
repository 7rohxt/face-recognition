import matplotlib.pyplot as plt
from firebase_configure import ref

# Get employee data from Firebase
data = ref.get()

# Storing name and attendance in a list
names = []
attendance_counts = []

for emp_id, details in data.items():
    names.append(details.get("name", emp_id))
    attendance_counts.append(details.get("total_attendance", 0))

# bar chart
plt.figure(figsize=(10, 6))
plt.bar(names, attendance_counts, color='skyblue')

plt.title("Employee Attendance Overview")
plt.xlabel("Employee")
plt.ylabel("Total Attendance Days")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  
plt.show()
