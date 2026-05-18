class TaskManager:
    def __init__(self):
        self.active_tasks = {}          # (cmd, target_id) -> task_id
        self.recent_tasks = {}          # (cmd, target_id) -> last_completed_time

    def submit_task(self, cmd, target_description, target_id):
        key = (cmd, target_id)
        # 若已有活跃任务，直接返回
        if key in self.active_tasks:
            return self.active_tasks[key], "already running"
        # 若刚完成（<10秒），跳过
        if key in self.recent_tasks and time.time() - self.recent_tasks[key] < 10:
            return None, "too soon, skip"
        # 创建新任务
        task_id = uuid4()
        self.active_tasks[key] = task_id
        # 执行任务...
        return task_id, "created"

CONFIDENCE_THRESHOLD = 0.80

def should_track(target_id, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(f"Target {target_id} confidence {confidence} < threshold, re-acquiring...")
        # 可选：尝试再次检测或融合其他传感器
        new_confidence = re_detect(target_id)
        if new_confidence < CONFIDENCE_THRESHOLD:
            return False, "confidence too low"
    return True, "ok"

class ReportAction:
    last_report_time = 0
    last_report_content = ""

    def execute(self, plan):
        now = time.time()
        content = plan.get('target_description', '')
        if (now - self.last_report_time < 5) and (content == self.last_report_content):
            return False, "duplicate report, skipped"
        # 执行报告（发送给用户或记录日志）
        self.last_report_time = now
        self.last_report_content = content
        return True, "reported"

# 任务ID复用（可选，可在mission_generator.py中实现类似逻辑）
# 例如：
# def get_or_create_task(cmd, target_id):
#     key = (cmd, target_id)
#     if key in task_id_map:
#         return task_id_map[key], "existing"
#     task_id = uuid4()
#     task_id_map[key] = task_id
#     return task_id, "created"