import psycopg2
import os

image_path = r"D:\subiksha projects\facial recognition\reference_faces"
if not os.path.exists(image_path):
    print("Error: File not found! Check the path:", image_path)
    exit()

image_name = os.path.basename(image_path)
# Connect to PostgreSQL
conn = psycopg2.connect(database="img1", user="postgres", password="postgresql", host="localhost", port="5432")
cur = conn.cursor()

# Read Image as Binary
with open(image_path, "rb") as image_file:
    binary_data = image_file.read()

# Insert into Database
cur.execute("INSERT INTO images (image_data) VALUES (%s)", (psycopg2.Binary(binary_data),))

# Commit and Close
conn.commit()

cur.execute("SELECT image_data FROM images WHERE id = 1")
image_data = cur.fetchone()[0]

# Save Retrieved Image
retrieved_path = "retrieved_image.jpg"
with open(retrieved_path, "wb") as image_file:
    image_file.write(image_data)

print(f"Image retrieved and saved as {retrieved_path}")
cur.close()
conn.close()
print("Image successfully stored and retrieved from the database.")