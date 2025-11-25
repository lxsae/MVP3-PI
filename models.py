from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), nullable=False, default='employee')  # admin, manager, employee
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'))
    position_id = db.Column(db.Integer, db.ForeignKey('position.id'))
    phone = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    department = db.relationship('Department', backref=db.backref('users', lazy=True))
    position = db.relationship('Position', backref=db.backref('users', lazy=True))
    attendances = db.relationship('Attendance', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == 'admin'

    def is_manager(self):
        return self.role in ['admin', 'manager']

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Position(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    check_in_time = db.Column(db.DateTime)
    check_out_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='present')  # present, late, absent
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    evidences = db.relationship('Evidence', backref='attendance', lazy=True, cascade='all, delete-orphan')

    @property
    def total_hours(self):
        if self.check_in_time and self.check_out_time:
            duration = self.check_out_time - self.check_in_time
            return duration.total_seconds() / 3600
        return 0

class Evidence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.id'), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # photo, note, checklist
    title = db.Column(db.String(128))
    content = db.Column(db.Text)  # For notes and checklist data
    file_path = db.Column(db.String(256))  # For uploaded files
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(64), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('logs', lazy=True))

    @staticmethod
    def log_action(user_id, action, details=None, ip_address=None):
        """Log a system action"""
        log = SystemLog(
            user_id=user_id,
            action=action,
            details=details,
            ip_address=ip_address
        )
        db.session.add(log)
        try:
            db.session.commit()
        except:
            db.session.rollback()

# Migration function to import existing CSV data
def migrate_csv_data(csv_file='asistencia_registros.csv'):
    """Migrate existing CSV data to database"""
    if not os.path.exists(csv_file):
        print("No CSV file found to migrate")
        return

    import csv
    from datetime import datetime

    migrated_count = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(row['Timestamp'])

                # Create or get user
                user = User.query.filter_by(
                    first_name=row['Nombre'],
                    last_name=row['Apellido']
                ).first()

                if not user:
                    user = User(
                        username=f"{row['Nombre'].lower()}.{row['Apellido'].lower()}",
                        email=row.get('Correo', f"{row['Nombre'].lower()}.{row['Apellido'].lower()}@company.com"),
                        first_name=row['Nombre'],
                        last_name=row['Apellido'],
                        role='employee'
                    )
                    user.set_password('default123')  # Default password, should be changed
                    db.session.add(user)
                    db.session.flush()  # Get user.id

                # Create attendance record
                attendance = Attendance(
                    user_id=user.id,
                    date=timestamp.date(),
                    check_in_time=timestamp,
                    status='present',
                    notes=f"Imported from CSV - Edad: {row.get('Edad', '')}, Sexo: {row.get('Sexo', '')}, Celular: {row.get('Celular', '')}"
                )
                db.session.add(attendance)
                migrated_count += 1

            except Exception as e:
                print(f"Error migrating row: {e}")
                continue

    try:
        db.session.commit()
        print(f"Successfully migrated {migrated_count} records from CSV")
    except Exception as e:
        db.session.rollback()
        print(f"Error during migration: {e}")