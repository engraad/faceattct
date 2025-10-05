# ==============================================================================
#                            مجدول المهام (Scheduler)
# ==============================================================================
# هذا الملف مسؤول عن تعريف وتشغيل المهام المجدولة في خلفية التطبيق.
# المهمة الرئيسية الحالية هي "تسجيل الغياب التلقائي" للموظفين.
# ==============================================================================

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, date, timedelta, time
import logging

# --- إعداد تسجيل الأنشطة (Logging) ---
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.INFO)

def mark_absent_employees():
    """
    [مهمة مجدولة] - يتم تشغيلها يوميًا لتسجيل الموظفين الذين لم يسجلوا حضورهم كـ "غائب".
    """
    # [إصلاح] تم نقل الاستيراد إلى هنا لكسر حلقة الاستيراد الدائري
    from app_new import app, db, EmployeeInfo, Attendance, SystemSetting

    # نستخدم سياق التطبيق (app_context) للسماح بالوصول إلى قاعدة البيانات خارج الطلبات العادية.
    with app.app_context():
        print(f"\n--- [{datetime.now()}] بدء المهمة اليومية: تسجيل الموظفين الغائبين ---")

        # الخطوة 1: الحصول على تاريخ اليوم وتحديد يوم الأسبوع (0 = الإثنين، 6 = الأحد).
        today = date.today()
        today_weekday = today.weekday()

        # الخطوة 2: جلب إعدادات النظام، وخاصة أيام عطلة نهاية الأسبوع.
        settings = SystemSetting.query.get(1)
        if not settings:
            print("⚠️ تم إحباط المهمة: لم يتم العثور على إعدادات النظام.")
            return

        # الخطوة 3: التحقق مما إذا كان اليوم هو يوم عطلة.
        try:
            # أيام العطلة مخزنة كنص مفصول بفاصلة، مثال: "4,5" (للجمعة والسبت).
            weekend_days = {int(day) for day in settings.weekend_days.split(',') if day.isdigit()}
            if today_weekday in weekend_days:
                print(f"ℹ️ اليوم هو عطلة نهاية الأسبوع (يوم {today_weekday}). لا يوجد إجراء.")
                return
        except (ValueError, TypeError, AttributeError) as e:
            # [تحسين] إذا كانت إعدادات الإجازة غير صحيحة، يتم إيقاف المهمة لمنع تسجيل الغياب بالخطأ
            print(f"❌ تم إيقاف المهمة: لا يمكن تحليل إعدادات أيام العطلة. يرجى مراجعة الإعدادات. الخطأ: {e}")
            return

        # الخطوة 4: جلب جميع الموظفين النشطين حاليًا في النظام.
        active_employees = EmployeeInfo.query.filter_by(status='active').all()
        if not active_employees:
            print("ℹ️ لا يوجد موظفون نشطون. انتهت المهمة.")
            return

        active_employee_ids = {emp.id for emp in active_employees}

        # الخطوة 5: جلب الموظفين الذين لديهم سجل حضور بالفعل لهذا اليوم (حاضر أو متأخر).
        records_today = Attendance.query.filter(
            Attendance.date == today.isoformat()
        ).all()
        
        checked_in_employee_ids = {rec.employee_info_id for rec in records_today}

        # الخطوة 6: تحديد الموظفين الغائبين عن طريق طرح الحاضرين من قائمة النشطين.
        absent_employee_ids = active_employee_ids - checked_in_employee_ids

        if not absent_employee_ids:
            print("✅ جميع الموظفين النشطين قد سجلوا حضورهم اليوم. لا يوجد غياب.")
            return

        print(f"🔍 تم العثور على {len(absent_employee_ids)} موظف غائب. جاري إنشاء السجلات...")

        # الخطوة 7: إنشاء سجلات "غياب" في قاعدة البيانات للموظفين الذين تم تحديدهم.
        absent_records_created = 0
        for employee_id in absent_employee_ids:
            try:
                new_absent_record = Attendance(
                    employee_info_id=employee_id,
                    date=today.isoformat(),
                    status='Absent', # تعيين الحالة إلى "غائب"
                    is_late=0 # غير مطبق هنا، ولكن يتم تعيينه إلى 0 للاتساق
                )
                db.session.add(new_absent_record)
                absent_records_created += 1
            except Exception as e:
                print(f"❌ فشل في إنشاء سجل غياب للموظف رقم {employee_id}: {e}")
                db.session.rollback() # التراجع عن التغييرات في حالة حدوث خطأ لهذا الموظف
                continue
        
        # حفظ جميع سجلات الغياب الجديدة في قاعدة البيانات.
        db.session.commit()
        print(f"✅ Successfully created {absent_records_created} 'Absent' records.")
        print("--- انتهت المهمة ---")

def initialize_scheduler():
    """
    يقوم بتهيئة وبدء المجدول الخلفي للمهام.
    """
    # [إصلاح] تم نقل الاستيراد إلى هنا لكسر حلقة الاستيراد الدائري
    from app_new import app, SystemSetting

    scheduler = BackgroundScheduler(daemon=True)
    
    # --- تحديد وقت تشغيل المهمة ---
    job_time_hour = 9
    job_time_minute = 35
    scheduled_job_time = None

    # محاولة قراءة الموعد النهائي من الإعدادات لتحديد وقت المهمة ديناميكيًا.
    try:
        with app.app_context():
            settings = SystemSetting.query.get(1)
            deadline_str = settings.attendance_deadline if (settings and settings.attendance_deadline) else '09:30:00'
            
            # [إصلاح] التعامل مع تنسيقات الوقت المختلفة (مع أو بدون الثواني)
            if len(deadline_str.split(':')) == 2:  # إذا كان التنسيق HH:MM فقط
                deadline_str += ':00'  # إضافة الثواني
            
            deadline_time = datetime.strptime(deadline_str, '%H:%M:%S').time()
            
            # [تحسين] جدولة المهمة لتعمل بعد دقيقة واحدة من الموعد النهائي.
            job_datetime = datetime.combine(date.today(), deadline_time) + timedelta(minutes=1)
            job_time_hour = job_datetime.hour
            job_time_minute = job_datetime.minute
            scheduled_job_time = job_datetime.time()

            print(f"[المجدول] سيتم تشغيل مهمة 'mark_absent_employees' يوميًا الساعة {job_time_hour:02d}:{job_time_minute:02d} بناءً على الإعدادات.")

    except Exception as e:
        print(f"[المجدول] ⚠️ لم نتمكن من قراءة الموعد النهائي من الإعدادات بسبب خطأ: {e}")
        print(f"[المجدول] سيتم استخدام الوقت الاحتياطي: {job_time_hour:02d}:{job_time_minute:02d}.")
        scheduled_job_time = time(job_time_hour, job_time_minute)

    # إضافة المهمة إلى المجدول لتعمل كل يوم في الساعة والدقيقة المحددة.
    scheduler.add_job(mark_absent_employees, 'cron', hour=job_time_hour, minute=job_time_minute)
    
    # [تحسين] التحقق مما إذا كان الوقت المجدول قد فات بالفعل لهذا اليوم.
    # إذا كان الأمر كذلك، يتم تشغيل المهمة مرة واحدة على الفور لتعويض ما فات.
    now_time = datetime.now().time()
    if scheduled_job_time and now_time > scheduled_job_time:
        print(f"[المجدول] ℹ️ الوقت المجدول ({scheduled_job_time.strftime('%H:%M')}) قد فات اليوم. سيتم تشغيل المهمة لمرة واحدة الآن.")
        # `run_date` : `None` تعني التشغيل الفوري
        scheduler.add_job(mark_absent_employees, 'date', run_date=None)

    # بدء تشغيل المجدول في الخلفية.
    scheduler.start()
    print("✅ تم بدء تشغيل مجدول المهام الخلفي بنجاح.")
    
    return scheduler