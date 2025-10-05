# Temporary script to clear attendance data
from app_new import app, db, Attendance

def clear_records():
    with app.app_context():
        try:
            num_deleted = db.session.query(Attendance).delete()
            db.session.commit()
            if num_deleted > 0:
                print(f"Successfully deleted {num_deleted} attendance records.")
            else:
                print("Attendance table is already empty.")
        except Exception as e:
            db.session.rollback()
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    clear_records()
