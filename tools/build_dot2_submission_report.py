#!/usr/bin/env python3
"""Build the official Round-2 progress report as print-ready HTML.

The stylesheet is reused from the earlier team draft. The body is generated from
the consolidated C1/C2 technical reports and the reproducible CarSky evidence in
this repository. Dates from the two workstream reports are intentionally not
used because their authors confirmed that those milestones were provisional.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLE_SOURCE = ROOT / "reports" / "Bao_Cao_Tien_Do_Dot_2_5ChangLinhNguLam_v2.html"
DEFAULT_OUTPUT = ROOT / "reports" / "Bao_Cao_Tien_Do_Dot_2_5ChangLinhNguLam.html"


def page(number: int | str, eyebrow: str, title: str, content: str, extra: str = "") -> str:
    return f"""
  <section class="page {extra}">
    <div class="topline">
      <div><span class="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
      <div class="section-index">{number}</div>
    </div>
    {content}
    <div class="footer">5 Chàng Lính Ngự Lâm · SafeLoop · Báo cáo tiến độ đợt 2</div>
    <div class="page-no">{number}</div>
  </section>"""


def simplify_v2_language(body: str) -> str:
    """Use plain Vietnamese for the BTC-facing v2, except AI/API terms."""
    replacements = [
        ("Driver Intelligence Platform<br>— SafeLoop", "Nền tảng hỗ trợ lái xe an toàn<br>— SafeLoop"),
        (
            "SafeLoop</strong> là lớp Driver Intelligence chạy trên Connected Car: nhận kết quả từ <strong>C1 Collision Intelligence</strong> và <strong>C2 Driver Intelligence</strong>",
            "SafeLoop</strong> là hệ thống hỗ trợ lái xe chạy trên nền tảng xe kết nối (Connected Car): nhận kết quả từ <strong>C1 dự đoán nguy cơ va chạm</strong> và <strong>C2 nhận biết trạng thái người lái</strong>",
        ),
        ("Dự đoán TTC từ video", "Dự đoán thời gian còn lại trước va chạm (TTC) từ video"),
        ("Fusion policy kết hợp", "Quy tắc kết hợp"),
        ("với safety gate cho microsleep", "với quy tắc an toàn cho microsleep"),
        ("ĐÃ ĐO OFFLINE", "ĐÃ ĐO TRÊN DỮ LIỆU LƯU SẴN"),
        ("ĐÃ ĐO REAL-TIME", "ĐÃ ĐO THEO THỜI GIAN THỰC"),
        ("MOCK/GT REPLAY", "MÔ PHỎNG / PHÁT LẠI NHÃN CHUẨN"),
        ("Product &amp; Integration Card", "Thông tin sản phẩm và cách tích hợp"),
        ("Fleet Manager", "người quản lý đội xe"),
        ("OEM/nhà cung cấp IVI", "hãng xe/nhà cung cấp màn hình trên xe (IVI)"),
        ("Core job", "Nhiệm vụ chính"),
        ("Input / Output", "Dữ liệu vào / Kết quả"),
        (
            "ego telemetry → TTC, driver state, contextual risk, hành động khuyến nghị, trip score",
            "dữ liệu vận hành xe → TTC, trạng thái người lái, mức rủi ro tổng hợp, hành động khuyến nghị, điểm chuyến đi",
        ),
        ("mô hình thật kết nối qua adapter đang là workstream kế tiếp", "bước tiếp theo là nối mô hình thật qua bộ chuyển đổi dữ liệu"),
        ("closed loop có explainability", "một quy trình khép kín có thể giải thích"),
        ("evidence chain", "chuỗi bằng chứng"),
        ("coaching sau chuyến", "gợi ý cải thiện sau chuyến đi"),
        ("hai candidate model", "hai mô hình thử nghiệm"),
        ("một product slice", "một bản demo thu gọn"),
        ("Ranh giới claim", "Phạm vi kết quả đã công bố"),
        (
            "product slice hiện dùng mock sinh động và T01 ground-truth replay. Chưa tuyên bố candidate model C1/C2 đã chạy end-to-end trên CarSky",
            "bản demo hiện dùng dữ liệu mô phỏng thay đổi theo thời gian và phát lại nhãn chuẩn của T01. Đội chưa công bố rằng mô hình C1/C2 thật đã chạy toàn bộ quy trình trên CarSky",
        ),
        ("PHẠM VI CAM KẾT TRONG PROPOSAL", "PHẠM VI ĐÃ CAM KẾT TRONG ĐỀ XUẤT"),
        ("Giữ nguyên outcome, triển khai theo lát cắt kiểm chứng", "Giữ nguyên mục tiêu, làm từng phần có thể kiểm chứng"),
        ("Cam kết proposal", "Cam kết trong đề xuất"),
        ("Evidence / khoảng trống", "Bằng chứng / phần còn thiếu"),
        ("C1: TTC / collision intelligence", "C1: thời gian còn lại trước va chạm (TTC)"),
        ("MODEL CANDIDATE", "MÔ HÌNH THỬ NGHIỆM"),
        ("đánh giá 6 sample", "đánh giá trên 6 bộ dữ liệu mẫu"),
        ("latency live", "thời gian xử lý theo thời gian thực"),
        ("cần tiếp tục domain-shift test", "cần tiếp tục kiểm thử domain shift"),
        ("Unified contextual risk", "Mức rủi ro tổng hợp theo tình huống"),
        ("PROTOTYPE", "BẢN THỬ NGHIỆM"),
        ("Rule-based fusion có reason/action", "Quy tắc kết hợp có nêu lý do và hành động"),
        ("Đã chạy mock và T01 GT; chưa calibrate bằng dữ liệu model thật", "Đã chạy bằng dữ liệu mô phỏng và nhãn chuẩn T01; chưa hiệu chỉnh bằng kết quả của mô hình thật"),
        ("Live dashboard / alert log", "Màn hình theo dõi trực tiếp / nhật ký cảnh báo"),
        ("PRODUCT SLICE", "BẢN DEMO THU GỌN"),
        ("Screen hiện phát trace nhúng; broker-to-screen bridge còn phải hoàn thiện", "Màn hình hiện phát dữ liệu mẫu được nhúng sẵn; cầu nối dữ liệu từ Broker đến màn hình vẫn cần hoàn thiện"),
        ("Post-trip score / coaching", "Điểm chuyến đi / gợi ý cải thiện"),
        ("C3 MOCK", "C3 MÔ PHỎNG"),
        ("Score/grade/penalty chạy xuyên trip", "Điểm, xếp hạng và mức trừ điểm được tính trong suốt chuyến đi"),
        ("fairness và ngưỡng coaching", "mức công bằng và ngưỡng đưa ra gợi ý"),
        ("CarSky deployment", "Triển khai trên CarSky"),
        ("Mock node → Central Broker đã verify 11/11 signals", "Nút mô phỏng → Central Broker đã kiểm tra đủ 11/11 tín hiệu"),
        ("Nguyên tắc thu hẹp scope", "Nguyên tắc giới hạn phạm vi"),
        ("Core flow để chấm", "Luồng chính dùng để trình bày"),
        ("Sample/Camera → C1+C2 → Fusion → VSS broker → AAOS Screen → trip evidence", "Dữ liệu mẫu/Camera → C1+C2 → kết hợp kết quả → VSS Broker → màn hình AAOS → bằng chứng chuyến đi"),
        ("Mọi output ngoài flow này chỉ được làm sau khi core flow lặp lại được", "Các kết quả ngoài luồng này chỉ được làm sau khi luồng chính có thể chạy lại ổn định"),
        ("Không claim ở đợt 2", "Những điều chưa công bố ở đợt 2"),
        ("Không claim production-ready, không claim C1 real-time, không claim model thật đã nối CarSky, không claim", "Chưa công bố sản phẩm đã sẵn sàng dùng thực tế, C1 đã chạy theo thời gian thực, mô hình thật đã nối CarSky hoặc"),
        ("telemetry contract VSS mở rộng", "quy ước dữ liệu VSS mở rộng"),
        ("mock fusion runtime", "chương trình mô phỏng kết hợp C1/C2"),
        ("C3 accumulator", "bộ tính điểm C3"),
        ("T01 GT replay generator", "công cụ phát lại nhãn chuẩn T01"),
        ("AAOS dashboard và bộ điều khiển cài đặt/verify CarSky", "màn hình AAOS và công cụ cài đặt/kiểm tra CarSky"),
        ("KIẾN TRÚC &amp; LUỒNG END-TO-END", "KIẾN TRÚC VÀ LUỒNG XỬ LÝ TOÀN BỘ HỆ THỐNG"),
        ("Một contract xuyên suốt từ inference đến màn hình", "Một quy ước dữ liệu chung từ kết quả AI đến màn hình"),
        ("DATA / REPLAY", "DỮ LIỆU / PHÁT LẠI"),
        ("Road video", "Video phía trước"),
        ("Cabin video", "Video trong cabin"),
        ("Ego telemetry", "Dữ liệu vận hành xe"),
        ("C3 · FUSION POLICY", "C3 · QUY TẮC KẾT HỢP"),
        ("contextual risk", "mức rủi ro tổng hợp"),
        ("reason + action", "lý do + hành động"),
        ("trip score", "điểm chuyến đi"),
        ("AAOS SCREEN", "MÀN HÌNH AAOS"),
        ("alert + evidence", "cảnh báo + bằng chứng"),
        ("EVIDENCE PLANE", "LỚP LƯU BẰNG CHỨNG"),
        ("contract/schema + run manifest + model/config hash + metrics + failure log + video", "quy ước/schema + thông tin lần chạy + mã băm model/cấu hình + chỉ số + nhật ký lỗi + video"),
        ("Replay đảm bảo cùng một sample có thể tái hiện model output, broker signal, alert và score", "Việc phát lại giúp cùng một bộ dữ liệu mẫu tái hiện được kết quả model, tín hiệu Broker, cảnh báo và điểm số"),
        ("Mục tiêu Code Freeze: artifact identity nối từ C1/C2 checkpoint đến từng frame và màn hình", "Mục tiêu khi chốt mã nguồn: mỗi checkpoint C1/C2 phải truy ra được từng khung hình và kết quả trên màn hình"),
        ("Contract tín hiệu đang dùng trên CarSky", "Quy ước tín hiệu đang dùng trên CarSky"),
        ("Ego</td>", "Dữ liệu xe</td>"),
        ("ĐÃ CHẠY: mock → broker", "ĐÃ CHẠY: mô phỏng → Broker"),
        ("ĐANG NỐI: model → adapter", "ĐANG LÀM: model → bộ chuyển đổi dữ liệu"),
        ("KẾ HOẠCH: broker → screen live", "KẾ HOẠCH: Broker → màn hình trực tiếp"),
        ("C1: candidate TTC có cải thiện rõ", "C1: mô hình TTC đã cải thiện rõ"),
        ("baseline mặc định", "mốc so sánh mặc định"),
        ("Thiết kế model và flow kỹ thuật", "Thiết kế model và luồng xử lý kỹ thuật"),
        ("bundle báo cáo 23 MB", "gói model có dung lượng 23 MB"),
        ("output giới hạn 10 s", "kết quả được giới hạn tối đa 10 s"),
        ("334 trips, 27,713 frames", "334 chuyến đi, 27,713 khung hình"),
        ("Artifact dự kiến bàn giao", "Tệp kỹ thuật dự kiến bàn giao"),
        ("notebook one-click", "notebook chạy bằng một lệnh"),
        ("đồ thị theo trip", "đồ thị theo chuyến đi"),
        ("Kết quả theo sample BTC", "Kết quả trên các bộ dữ liệu mẫu của BTC"),
        ("bằng chứng offline, chưa phải pipeline real-time", "kết quả đo trên dữ liệu lưu sẵn, chưa phải quy trình thời gian thực"),
        ("C2: pipeline causal vượt tốc độ video thời gian thực", "C2: pipeline causal đủ tốc độ xử lý video theo thời gian thực"),
        ("Median latency", "Độ trễ trung vị"),
        ("Peak VRAM", "VRAM cao nhất"),
        ("đủ confidence", "đủ độ tin cậy"),
        ("safety gate quyết định", "quy tắc an toàn quyết định"),
        ("không nhìn future frame", "không dùng khung hình tương lai"),
        ("state được reset giữa các trip", "trạng thái được đặt lại giữa các chuyến đi"),
        ("Case khó cần phân tích", "Trường hợp khó cần phân tích"),
        ("Rủi ro model", "Rủi ro của model"),
        ("góc mặt nghiêng gây false closure", "góc mặt nghiêng làm hệ thống nhận nhầm mắt nhắm"),
        ("Mitigation:", "Cách xử lý:"),
        ("CarSky product slice: đã deploy, có signal live và màn hình AAOS", "Bản demo CarSky: đã triển khai, có tín hiệu trực tiếp và màn hình AAOS"),
        ("CarSky deployment", "Lần triển khai CarSky"),
        ("VSS signals có mặt", "Có đủ tín hiệu VSS"),
        ("verify không thiếu path", "không thiếu đường dẫn tín hiệu"),
        ("observer updates", "lần cập nhật quan sát được"),
        ("automated tests pass", "kiểm thử tự động đạt"),
        ("local integration repo", "kho mã tích hợp trên máy"),
        ("Blueprint <strong>SafeLoop Mock Integration Lab</strong>", "Sơ đồ triển khai <strong>SafeLoop Mock Integration Lab</strong>"),
        ("Deployment</td>", "Lần triển khai</td>"),
        ("Mock C1/C2 Fusion → Central Broker", "Mô phỏng kết hợp C1/C2 → Central Broker"),
        ("T01 sample GT replay", "Phát lại nhãn chuẩn của mẫu T01"),
        ("REPEATABLE", "CÓ THỂ LẶP LẠI"),
        ("VISIBLE", "ĐÃ HIỂN THỊ"),
        ("11 VSS paths; TTC, distance, speed và acceleration thay đổi", "11 đường dẫn VSS; TTC, khoảng cách, tốc độ và gia tốc thay đổi"),
        ("600 source frames @20 Hz; dashboard compact 151 samples @5 Hz", "600 khung hình nguồn @20 Hz; màn hình dùng 151 điểm dữ liệu @5 Hz"),
        ("GT replay minh bạch, không phải output model live", "phát lại nhãn chuẩn, không phải kết quả của model chạy theo thời gian thực"),
        ("Probe data-plane", "Kiểm tra luồng dữ liệu"),
        ("update cho ba signal ego", "lần cập nhật cho ba tín hiệu vận hành xe"),
        ("REST replay", "phát dữ liệu qua REST"),
        ("không coi đây là lỗi product", "không coi đây là lỗi của sản phẩm"),
        ("Kịch bản 8 phút: claim nào cũng có evidence", "Kịch bản 8 phút: nội dung nào cũng có bằng chứng"),
        ("Problem &amp; outcome", "Vấn đề và kết quả mong muốn"),
        ("hợp nhất hai context", "kết hợp hai nguồn thông tin"),
        ("C1 evidence", "Kết quả C1"),
        ("Mở evaluation của 6 sample", "Mở kết quả đánh giá trên 6 bộ dữ liệu mẫu"),
        ("case T05", "trường hợp T05"),
        ("C2 evidence", "Kết quả C2"),
        ("Chạy/chiếu one-click evaluation", "Chạy hoặc trình chiếu bài đánh giá bằng một lệnh"),
        ("CarSky runtime", "Trạng thái CarSky khi chạy"),
        ("T01 core story", "Tình huống chính T01"),
        ("driver distracted", "người lái mất tập trung"),
        ("pedestrian jaywalk", "người đi bộ băng qua đường"),
        ("cảnh báo chuyển CRITICAL", "cảnh báo chuyển sang NGUY HIỂM"),
        ("Failure/limitation", "Giới hạn"),
        ("model chưa nối platform", "model chưa nối với CarSky"),
        ("chỉ ra fallback", "nêu phương án dự phòng"),
        ("Close. Chốt product value, workstream còn lại và evidence package", "Kết luận. Chốt giá trị sản phẩm, việc còn lại và bộ bằng chứng"),
        ("Runbook demo T01 trên CarSky", "Các bước demo T01 trên CarSky"),
        ("sinh payload bằng", "tạo gói lệnh bằng"),
        ("Expected proof", "Kết quả cần quan sát"),
        ("deployment identity", "mã lần triển khai"),
        ("signal values đổi qua ít nhất ba sample", "giá trị tín hiệu thay đổi qua ít nhất ba lần đọc"),
        ("score/penalty tích lũy", "điểm và mức trừ điểm được cộng dồn"),
        ("log/manifest khớp artifact", "nhật ký và danh sách tệp khớp với tệp kỹ thuật"),
        ("Fallback có kiểm soát", "Phương án dự phòng"),
        ("dashboard local cùng trace T01", "màn hình trên máy với cùng dữ liệu T01"),
        ("video “giả live”", "video giả làm dữ liệu đang chạy"),
        ("Dashboard KPI: số đã đo, nguồn và mức tin cậy", "Tổng hợp KPI: kết quả đo, nguồn và mức tin cậy"),
        ("6 samples", "6 bộ dữ liệu mẫu"),
        ("default 37.2", "mốc mặc định 37.2"),
        ("C1 runtime", "Thời gian xử lý C1"),
        ("225 windows", "225 cửa sổ dữ liệu"),
        ("C2 performance", "Tốc độ xử lý C2"),
        ("6/6 sample BTC", "6/6 bộ dữ liệu mẫu của BTC"),
        ("60/60 observer updates", "60/60 lần cập nhật quan sát được"),
        ("20 Hz probe, 3 ego signals", "Kiểm tra ở 20 Hz với 3 tín hiệu vận hành xe"),
        ("Mock fusion signal coverage", "Mức độ đầy đủ của tín hiệu mô phỏng"),
        ("no missing path", "không thiếu đường dẫn tín hiệu"),
        ("Integration test suite", "Bộ kiểm thử tích hợp"),
        ("90 passed", "90 kiểm thử đạt"),
        ("Local repo snapshot", "Kho mã trên máy tại thời điểm"),
        ("T01 scenario", "Kịch bản T01"),
        ("Ground-truth sample replay", "Phát lại nhãn chuẩn của dữ liệu mẫu"),
        ("TEAM REPORT", "BÁO CÁO NHÓM KỸ THUẬT"),
        ("NOT LIVE", "CHƯA THEO THỜI GIAN THỰC"),
        ("RAW JSON", "TỆP JSON GỐC"),
        ("REPRODUCED", "ĐÃ CHẠY LẠI"),
        ("DETERMINISTIC", "KẾT QUẢ CỐ ĐỊNH"),
        ("Acceptance gates trước Code Freeze", "Các điều kiện cần đạt trước khi chốt mã nguồn"),
        ("Gate</th>", "Mốc kiểm tra</th>"),
        ("G1 · Reproducibility", "G1 · Có thể chạy lại"),
        ("Clean run tạo đúng CSV/metrics/manifest", "Chạy lại từ đầu tạo đúng CSV, chỉ số và danh sách tệp"),
        ("Có report; cần gom code, checkpoint và hash", "Đã có báo cáo; cần gom mã nguồn, checkpoint và mã băm"),
        ("G2 · E2E product", "G2 · Toàn bộ quy trình sản phẩm"),
        ("Ít nhất một sample chạy model thật → fusion → CarSky → Screen", "Ít nhất một bộ dữ liệu mẫu chạy qua model thật → kết hợp kết quả → CarSky → màn hình"),
        ("Chưa đạt; hiện mock/GT", "Chưa đạt; hiện dùng mô phỏng/nhãn chuẩn"),
        ("G3 · Latency", "G3 · Độ trễ"),
        ("C1 có profile và mode demo phù hợp", "C1 có số đo thời gian xử lý và cách demo phù hợp"),
        ("C1 chưa đạt live", "C1 chưa đạt thời gian thực"),
        ("G4 · Robustness", "G4 · Khả năng xử lý lỗi"),
        ("Reset trip, missing/null input, broker restart có expected behavior", "Đặt lại chuyến đi, dữ liệu thiếu/rỗng và khởi động lại Broker đều cho kết quả đúng dự kiến"),
        ("Một phần đã test; cần fault matrix E2E", "Đã kiểm tra một phần; cần bảng kiểm thử lỗi cho toàn bộ quy trình"),
        ("G5 · Evidence identity", "G5 · Bằng chứng đồng nhất"),
        ("Commit + config + model hash + run ID + video khớp nhau", "Commit + cấu hình + mã băm model + mã lần chạy + video phải khớp nhau"),
        ("Platform có; model artifacts cần nhập repo", "CarSky đã có; các tệp model cần đưa vào kho mã"),
        ("Đường găng không nằm ở UI mà ở integration contract", "Ưu tiên lớn nhất là kết nối đúng model với CarSky, không phải làm đẹp giao diện"),
        ("Mitigation / quyết định", "Cách xử lý / quyết định"),
        ("Không thể claim live collision model", "Chưa thể công bố mô hình va chạm chạy theo thời gian thực"),
        ("window lag + T4 runtime cao", "độ trễ cửa sổ + thời gian xử lý trên T4 còn cao"),
        ("Profile, FP16/batch", "Đo chi tiết thời gian xử lý, dùng FP16/batch"),
        ("chế độ offline replay", "chế độ phát lại dữ liệu lưu sẵn"),
        ("Mentor góp ý trade-off metric–latency và target compute trên xe", "Đề nghị mentor góp ý cách cân bằng độ chính xác, độ trễ và cấu hình phần cứng trên xe"),
        ("Xác nhận protocol/label semantics", "Xác nhận quy trình và ý nghĩa nhãn"),
        ("Model artifacts phân tán", "Các tệp model đang nằm rải rác"),
        ("Không tái lập được claim từ integration repo", "Chưa thể chạy lại kết quả từ kho mã tích hợp"),
        ("Artifact contract, checksum, lockfile, one-click command, shared evidence index", "Quy chuẩn tệp bàn giao, checksum, lockfile, lệnh chạy một bước và mục lục bằng chứng chung"),
        ("các workstream kỹ thuật cần bàn giao đúng gate", "các nhóm kỹ thuật cần bàn giao đúng mốc kiểm tra"),
        ("Broker → Screen chưa live", "Broker chưa gửi dữ liệu trực tiếp đến màn hình"),
        ("Screen đẹp nhưng chưa phản ánh signal runtime", "Màn hình chưa hiển thị tín hiệu đang chạy từ Broker"),
        ("Xây bridge subscriber nhỏ", "Xây cầu nối nhỏ để đọc dữ liệu"),
        ("timestamp/sequence/health indicator; stale-data fallback", "dấu thời gian, số thứ tự, trạng thái kết nối và phương án khi dữ liệu quá cũ"),
        ("CarSky guidance", "Hướng dẫn từ CarSky"),
        ("automation Screen hạn chế", "khả năng tự động điều khiển màn hình bị hạn chế"),
        ("lưu payload/runbook", "lưu gói lệnh và hướng dẫn chạy"),
        ("giới hạn tenant", "giới hạn của không gian dự án"),
        ("Safety / privacy", "An toàn / quyền riêng tư"),
        ("Over-warning", "Cảnh báo quá nhiều"),
        ("Fail-silent, advisory only, local feature extraction", "Khi lỗi thì không phát cảnh báo sai, chỉ đưa ra khuyến nghị và trích xuất đặc trưng ngay trên thiết bị"),
        ("audit trail", "nhật ký kiểm tra"),
        ("Mentor review HMI escalation và data-retention", "Đề nghị mentor xem lại mức cảnh báo trên giao diện HMI và thời gian lưu dữ liệu"),
        ("Failure paths đã khảo sát trên CarSky", "Các tình huống lỗi đã kiểm tra trên CarSky"),
        ("Payload / schema", "Dữ liệu gửi lên / schema"),
        ("Unknown signal", "Tín hiệu không tồn tại"),
        ("batch có thể không atomic", "một lô dữ liệu có thể không được xử lý toàn vẹn"),
        ("Cyclic route không được hỗ trợ", "Đường kết nối tạo thành vòng lặp không được hỗ trợ"),
        ("Runtime / API", "Khi chạy / API"),
        ("Runtime route có sai khác theo API surface", "Đường dẫn API khi chạy có khác nhau giữa các nhóm API"),
        ("Conduit missing", "Thiếu dịch vụ Conduit"),
        ("Credential format và method mismatch", "Sai định dạng credential và phương thức API"),
        ("stale/missing/schema mismatch", "quá cũ, bị thiếu hoặc sai schema"),
        ("Năm gate theo dependency, không khóa vào mốc ngày sai", "Năm mốc công việc theo thứ tự phụ thuộc, không dùng các ngày chưa chắc chắn"),
        ("Gate 1", "Mốc 1"),
        ("Gate 2", "Mốc 2"),
        ("Gate 3", "Mốc 3"),
        ("Gate 4", "Mốc 4"),
        ("Gate 5", "Mốc 5"),
        ("Artifact Freeze C1/C2", "Chốt tệp kỹ thuật C1/C2"),
        ("dependency lock", "danh sách phiên bản thư viện"),
        ("lệnh reproduce vào cùng submission snapshot", "lệnh chạy lại vào cùng bộ hồ sơ nộp"),
        ("Inference Contract", "Quy ước kết quả của model"),
        ("missing-data semantics và output schema dùng chung", "cách xử lý dữ liệu thiếu và schema kết quả dùng chung"),
        ("khóa một mode chính thức sau profile", "chọn một cách chạy chính thức sau khi đo thời gian xử lý"),
        ("live tối ưu hoặc offline/replay được công bố", "thời gian thực đã tối ưu hoặc phát lại dữ liệu lưu sẵn và công bố rõ"),
        ("Broker → Screen live", "Broker → màn hình trực tiếp"),
        ("Thay trace nhúng bằng subscriber bridge", "Thay dữ liệu nhúng sẵn bằng cầu nối đọc dữ liệu từ Broker"),
        ("hiển thị health, age và source mode MODEL/MOCK/GT", "hiển thị trạng thái kết nối, độ cũ của dữ liệu và nguồn MODEL/MÔ PHỎNG/NHÃN CHUẨN"),
        ("Evidence Freeze", "Chốt bộ bằng chứng"),
        ("fault matrix, latency", "bảng kiểm thử lỗi, độ trễ"),
        ("khóa Claim–Evidence Map và dry-run Q&amp;A", "chốt bảng nội dung–bằng chứng và diễn tập phần hỏi đáp"),
        ("ĐƯỜNG GĂNG", "VIỆC QUAN TRỌNG"),
        ("Definition of Done", "Điều kiện hoàn thành"),
        ("Workstream", "Hạng mục"),
        ("6 sample reproducible; report khớp bundle hash; latency profile có quyết định mode", "6 bộ dữ liệu mẫu chạy lại được; báo cáo khớp mã băm gói model; có số đo độ trễ để chọn cách chạy"),
        ("6 sample + causal/reset tests; checkpoint/schema fingerprint; adapter phát output thật", "6 bộ dữ liệu mẫu + kiểm thử causal/reset; checkpoint khớp schema; bộ chuyển đổi phát kết quả thật"),
        ("Policy versioned; reason/action deterministic; test edge cases và fairness", "Quy tắc có phiên bản; lý do/hành động cho kết quả cố định; kiểm thử trường hợp đặc biệt và mức công bằng"),
        ("Score “magic number” không giải thích", "Điểm số không có công thức giải thích"),
        ("restart/replay lặp lại được", "khởi động lại/phát lại vẫn cho kết quả ổn định"),
        ("Screen độc lập với signal runtime", "Màn hình không nhận tín hiệu đang chạy"),
        ("Submission", "Bộ hồ sơ nộp"),
        ("Commit/artifact/config/video/claim map cùng một identity", "Commit, tệp kỹ thuật, cấu hình, video và bảng nội dung–bằng chứng phải cùng một phiên bản"),
        ("khóa reproducibility và model-to-platform trước; polishing UI, fleet map và coaching nâng cao chỉ thực hiện sau Gate 4", "ưu tiên khả năng chạy lại và kết nối model với CarSky; chỉ làm đẹp giao diện, bản đồ đội xe và gợi ý nâng cao sau Mốc 4"),
        ("02 · Phạm vi cam kết trong proposal", "02 · Phạm vi đã cam kết trong đề xuất"),
        ("03 · Kiến trúc & luồng end-to-end", "03 · Kiến trúc và luồng xử lý toàn bộ hệ thống"),
        ("VSS path tiêu biểu", "Đường dẫn VSS tiêu biểu"),
        ("hai mô hình thử nghiệm đã có kết quả trên 6 sample", "hai mô hình thử nghiệm đã có kết quả trên 6 bộ dữ liệu mẫu"),
        ("Bundle lai V-JEPA2", "Gói model kết hợp V-JEPA2"),
        ("Pipeline causal, reset theo trip", "Pipeline causal, đặt lại trạng thái sau mỗi chuyến đi"),
        ("Blueprint hợp lệ, deployment RUNNING", "Sơ đồ triển khai hợp lệ, lần triển khai ở trạng thái RUNNING"),
        ("<th>Trip</th>", "<th>Chuyến</th>"),
        ("16 frame tạo độ trễ", "16 khung hình tạo độ trễ"),
        ("cho trip 30 s", "cho chuyến đi 30 s"),
        ("3,600 frames", "3,600 khung hình"),
        ("input 640×384", "đầu vào 640×384"),
        ("LIVE</span>", "TRỰC TIẾP</span>"),
        ("C3 score", "điểm C3"),
        ("Hình 1 — Screen T01-SAMPLE hiển thị C1, C2, fusion và C3", "Hình 1 — Màn hình T01-SAMPLE hiển thị C1, C2, phần kết hợp kết quả và C3"),
        ("trong probe 5 frame", "trong lần kiểm tra 5 khung hình"),
        ("OFFLINE</span>", "DỮ LIỆU LƯU SẴN</span>"),
        ("Mở deployment RUNNING", "Mở lần triển khai đang ở trạng thái RUNNING"),
        ("Nói rõ C1 chưa real-time", "Nói rõ C1 chưa chạy theo thời gian thực"),
        ("khi path lỗi hoặc payload null", "khi đường dẫn lỗi hoặc gói dữ liệu rỗng"),
        ("HONEST</span>", "MINH BẠCH</span>"),
        ("<strong>Close.</strong>", "<strong>Kết luận.</strong>"),
        ("Chốt product value, workstream còn lại và evidence package", "Chốt giá trị sản phẩm, việc còn lại và bộ bằng chứng"),
        ("Q&amp;A</span>", "HỎI ĐÁP</span>"),
        ("Kiểm tra deployment:", "Kiểm tra lần triển khai:"),
        ("Nếu cần tạo lại từ sample:", "Nếu cần tạo lại từ bộ dữ liệu mẫu:"),
        ("screen chuyển trạng thái", "màn hình chuyển trạng thái"),
        ("nếu Screen/ADB lỗi", "nếu màn hình/ADB lỗi"),
        ("165–168 s / trip 30 s", "165–168 s / chuyến đi 30 s"),
        ("(3,600 frames)", "(3,600 khung hình)"),
        ("11/11 paths", "11/11 đường dẫn"),
        ("Deployment RUNNING", "Lần triển khai RUNNING"),
        ("600 frames", "600 khung hình"),
        ("C1 chưa real-time", "C1 chưa chạy theo thời gian thực"),
        ("Đề nghị mentor", "Đề nghị cố vấn"),
        ("raw video", "video gốc"),
        ("Chạy 6 sample", "Chạy 6 bộ dữ liệu mẫu"),
        ("Đưa code, checkpoint/bundle, config", "Đưa mã nguồn, checkpoint/gói model, tệp cấu hình"),
        ("đã đạt real-time", "đã đạt tốc độ thời gian thực"),
        ("Model output xuất hiện", "Kết quả model xuất hiện"),
        ("Chỉ có số trong slide/DOCX", "Chỉ có số trong trang trình chiếu/DOCX"),
        ("khóa reproducibility và model-to-platform trước; polishing UI, fleet map và coaching nâng cao chỉ thực hiện sau Mốc 4", "ưu tiên khả năng chạy lại và kết nối model với CarSky; chỉ làm đẹp giao diện, bản đồ đội xe và gợi ý nâng cao sau Mốc 4"),
    ]
    for old, new in replacements:
        body = body.replace(old, new)
    return body


def build_body(compact_v2: bool = False) -> str:
    sections: list[str] = []
    cover_identity = (
        "<div><strong>Sản phẩm</strong>SafeLoop</div>"
        if compact_v2
        else "<div><strong>Trưởng nhóm</strong>Nguyễn Thanh Sơn</div>"
    )

    sections.append(f"""
  <section class="page cover">
    <span class="eyebrow">FPT Automotive Hackathon 2026 · Báo cáo tiến độ đợt 2</span>
    <h1>Driver Intelligence Platform<br>— SafeLoop</h1>
    <p class="subtitle">Hợp nhất rủi ro va chạm, trạng thái người lái và điểm an toàn chuyến đi trên nền tảng CarSky.</p>
    <div class="one-liner">Không chỉ phát hiện nguy cơ: SafeLoop biến hai luồng AI thành một quyết định an toàn có giải thích, có bằng chứng và có đường triển khai lên xe.</div>
    <div class="cover-meta">
      <div><strong>Đội thi</strong>5 Chàng Lính Ngự Lâm</div>
      {cover_identity}
      <div><strong>Bản báo cáo</strong>04/08/2026 · v2.0</div>
    </div>
  </section>""")

    sections.append(page(1, "01 · Tóm tắt giải pháp", "Một sản phẩm, ba tầng giá trị", """
    <p class="lede"><strong>SafeLoop</strong> là lớp Driver Intelligence chạy trên Connected Car: nhận kết quả từ <strong>C1 Collision Intelligence</strong> và <strong>C2 Driver Intelligence</strong>, hợp nhất theo ngữ cảnh thành cảnh báo ưu tiên, đồng thời tích lũy <strong>C3 Safe Driving Score</strong> cho tài xế và đội xe.</p>
    <div class="grid-3">
      <div class="card blue"><h3>C1 · Nhìn ra đường</h3><p>Dự đoán TTC từ video bằng V-JEPA2 và head lai classification–regression; ưu tiên độ chính xác vùng TTC nguy hiểm.</p><span class="badge provided">ĐÃ ĐO OFFLINE</span></div>
      <div class="card green"><h3>C2 · Nhìn người lái</h3><p>MobileNetV3-Large + hai nhánh LSTM causal, xuất một trong năm trạng thái với safety gate cho microsleep.</p><span class="badge verified">ĐÃ ĐO REAL-TIME</span></div>
      <div class="card amber"><h3>C3 · Ra quyết định</h3><p>Fusion policy kết hợp TTC, khoảng cách, distraction/fatigue và điểm chuyến đi để chọn cảnh báo có lý do.</p><span class="badge progress">MOCK/GT REPLAY</span></div>
    </div>
    <h3>Product &amp; Integration Card</h3>
    <table class="tight">
      <tr><th>Người dùng</th><td>Tài xế, Fleet Manager, OEM/nhà cung cấp IVI; mở rộng cho bảo hiểm theo sự đồng ý của người dùng.</td></tr>
      <tr><th>Core job</th><td>Phát hiện sớm tình huống “nguy cơ phía trước + tài xế chưa sẵn sàng”, ưu tiên cảnh báo và ghi lại bằng chứng sau chuyến.</td></tr>
      <tr><th>Input / Output</th><td>Video đường, video cabin, ego telemetry → TTC, driver state, contextual risk, hành động khuyến nghị, trip score.</td></tr>
      <tr><th>Điểm tích hợp</th><td>CarSky Central Broker (VSS), Script Node, Android Automotive Screen/WebView; mô hình thật kết nối qua adapter đang là workstream kế tiếp.</td></tr>
      <tr><th>Giá trị khác biệt</th><td>Hợp nhất C1+C2 thành closed loop có explainability; cùng một evidence chain phục vụ cảnh báo tức thời và coaching sau chuyến.</td></tr>
    </table>
    <div class="callout success"><strong>Giá trị đã chứng minh ở giữa kỳ:</strong> hai candidate model đã có kết quả trên 6 sample; một product slice C1+C2+C3 đã chạy trên CarSky và hiển thị trên màn hình Android Automotive.</div>
    <div class="callout warning"><strong>Ranh giới claim:</strong> product slice hiện dùng mock sinh động và T01 ground-truth replay. Chưa tuyên bố candidate model C1/C2 đã chạy end-to-end trên CarSky.</div>
    """))

    sections.append(page(2, "02 · Phạm vi cam kết trong proposal", "Giữ nguyên outcome, triển khai theo lát cắt kiểm chứng", """
    <table>
      <thead><tr><th>Cam kết proposal</th><th>Hiện trạng đợt 2</th><th>Evidence / khoảng trống</th></tr></thead>
      <tbody>
        <tr><td>C1: TTC / collision intelligence</td><td><span class="badge provided">MODEL CANDIDATE</span><br>Bundle lai V-JEPA2; đánh giá 6 sample.</td><td>Composite 53.2; MAE Critical 1.780 s. Chưa đạt latency live trên T4.</td></tr>
        <tr><td>C2: 5 trạng thái người lái</td><td><span class="badge verified">MODEL CANDIDATE</span><br>Pipeline causal, reset theo trip.</td><td>Composite 81.1; 27.65 FPS trên T4; cần tiếp tục domain-shift test.</td></tr>
        <tr><td>Unified contextual risk</td><td><span class="badge progress">PROTOTYPE</span><br>Rule-based fusion có reason/action.</td><td>Đã chạy mock và T01 GT; chưa calibrate bằng dữ liệu model thật.</td></tr>
        <tr><td>Live dashboard / alert log</td><td><span class="badge progress">PRODUCT SLICE</span><br>AAOS 1920×1080; C1/C2/C3 cùng màn hình.</td><td>Screen hiện phát trace nhúng; broker-to-screen bridge còn phải hoàn thiện.</td></tr>
        <tr><td>Post-trip score / coaching</td><td><span class="badge progress">C3 MOCK</span><br>Score/grade/penalty chạy xuyên trip.</td><td>Cần khóa công thức, fairness và ngưỡng coaching.</td></tr>
        <tr><td>CarSky deployment</td><td><span class="badge verified">RUNNING</span><br>Blueprint hợp lệ, deployment RUNNING.</td><td>Mock node → Central Broker đã verify 11/11 signals.</td></tr>
      </tbody>
    </table>
    <h3>Nguyên tắc thu hẹp scope</h3>
    <div class="grid-2">
      <div class="card blue"><h3>Core flow để chấm</h3><p><strong>Sample/Camera → C1+C2 → Fusion → VSS broker → AAOS Screen → trip evidence.</strong> Mọi output ngoài flow này chỉ được làm sau khi core flow lặp lại được.</p></div>
      <div class="card soft"><h3>Không claim ở đợt 2</h3><p>Không claim production-ready, không claim C1 real-time, không claim model thật đã nối CarSky, không claim chứng nhận an toàn hay can thiệp điều khiển xe.</p></div>
    </div>
    <div class="callout"><strong>Điểm mới do đội tự phát triển:</strong> telemetry contract VSS mở rộng, mock fusion runtime, C3 accumulator, T01 GT replay generator, AAOS dashboard và bộ điều khiển cài đặt/verify CarSky.</div>
    """))

    sections.append(page(3, "03 · Kiến trúc & luồng end-to-end", "Một contract xuyên suốt từ inference đến màn hình", """
    <div class="architecture">
      <svg viewBox="0 0 1000 470" role="img" aria-label="SafeLoop end-to-end architecture">
        <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#1266e3"/></marker></defs>
        <rect x="20" y="42" width="180" height="150" rx="16" fill="#edf5ff" stroke="#8eb9f4"/><text x="40" y="72" font-weight="700">DATA / REPLAY</text><text x="40" y="105">Road video</text><text x="40" y="132">Cabin video</text><text x="40" y="159">Ego telemetry</text>
        <rect x="270" y="20" width="210" height="86" rx="16" fill="#edf5ff" stroke="#8eb9f4"/><text x="290" y="52" font-weight="700">C1 · TTC</text><text x="290" y="79">V-JEPA2 + hybrid head</text>
        <rect x="270" y="126" width="210" height="100" rx="16" fill="#eaf8f2" stroke="#83cfaf"/><text x="290" y="157" font-weight="700">C2 · DRIVER STATE</text><text x="290" y="184">MobileNetV3 + LSTM</text><text x="290" y="207">causal safety gates</text>
        <rect x="540" y="63" width="200" height="126" rx="16" fill="#fff6df" stroke="#e2b960"/><text x="560" y="94" font-weight="700">C3 · FUSION POLICY</text><text x="560" y="122">contextual risk</text><text x="560" y="149">reason + action</text><text x="560" y="174">trip score</text>
        <rect x="800" y="20" width="180" height="90" rx="16" fill="#f5f8fb" stroke="#9aacbe"/><text x="820" y="52" font-weight="700">CARSKY</text><text x="820" y="79">Central Broker / VSS</text>
        <rect x="800" y="130" width="180" height="90" rx="16" fill="#eaf8f2" stroke="#83cfaf"/><text x="820" y="162" font-weight="700">AAOS SCREEN</text><text x="820" y="189">alert + evidence</text>
        <line x1="200" y1="88" x2="270" y2="66" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/><line x1="200" y1="146" x2="270" y2="174" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/><line x1="480" y1="66" x2="540" y2="105" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/><line x1="480" y1="176" x2="540" y2="147" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/><line x1="740" y1="105" x2="800" y2="68" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/><line x1="890" y1="110" x2="890" y2="130" stroke="#1266e3" stroke-width="3" marker-end="url(#arr)"/>
        <rect x="20" y="288" width="960" height="138" rx="16" fill="#f8fbfe" stroke="#cbd8e6"/><text x="45" y="320" font-weight="700">EVIDENCE PLANE</text><text x="45" y="350">contract/schema + run manifest + model/config hash + metrics + failure log + video</text><text x="45" y="382">Replay đảm bảo cùng một sample có thể tái hiện model output, broker signal, alert và score.</text><text x="45" y="409" fill="#52657d">Mục tiêu Code Freeze: artifact identity nối từ C1/C2 checkpoint đến từng frame và màn hình.</text>
      </svg>
    </div>
    <h3>Contract tín hiệu đang dùng trên CarSky</h3>
    <table class="tight">
      <tr><th>Nhóm</th><th>VSS path tiêu biểu</th><th>Ý nghĩa</th></tr>
      <tr><td>Ego</td><td><code>Vehicle.Speed</code>, <code>Acceleration.Longitudinal/Lateral</code></td><td>Ngữ cảnh động học.</td></tr>
      <tr><td>C1</td><td><code>...ObstacleDetection.Front.Center.TimeGap</code>, <code>Distance</code>, <code>IsWarning</code></td><td>TTC ms, khoảng cách m, cảnh báo va chạm.</td></tr>
      <tr><td>C2</td><td><code>Vehicle.Driver.AttentiveProbability</code>, <code>DistractionLevel</code>, <code>FatigueLevel</code>, <code>IsEyesOnRoad</code>, <code>Vehicle.ADAS.DMS.IsWarning</code></td><td>Trạng thái và mức độ rủi ro cabin.</td></tr>
    </table>
    <div class="legend"><span class="badge verified">ĐÃ CHẠY: mock → broker</span> <span class="badge progress">ĐANG NỐI: model → adapter</span> <span class="badge planned">KẾ HOẠCH: broker → screen live</span></div>
    """))

    sections.append(page(4, "04 · Kết quả đã hoàn thành", "C1: candidate TTC có cải thiện rõ ở vùng nguy hiểm", """
    <div class="grid-4">
      <div class="metric"><div class="value">53.2</div><div class="label">Composite / 6 samples<br>baseline mặc định: 37.2</div></div>
      <div class="metric"><div class="value">1.780s</div><div class="label">MAE Critical<br>giảm 66.6% từ 5.338s</div></div>
      <div class="metric"><div class="value">0.566</div><div class="label">F1 tại TTC &lt; 2s<br>baseline: 0.600</div></div>
      <div class="metric"><div class="value">0.3809</div><div class="label">Inverse-TTC MAE<br>cải thiện 19.4%</div></div>
    </div>
    <h3>Thiết kế model và flow kỹ thuật</h3>
    <div class="flow">
      <div class="node"><strong>16 RGB frames</strong>640×360 · 8 FPS<br>stride 1</div><div class="arrow">→</div>
      <div class="node"><strong>V-JEPA2</strong>ViT-L backbone<br>đóng băng</div><div class="arrow">→</div>
      <div class="node"><strong>Hybrid heads</strong>calibrated collision<br>+ TTC regression</div><div class="arrow">→</div>
      <div class="node"><strong>Temporal output</strong>blend w=0.25<br>median window 11</div>
    </div>
    <ul>
      <li>Backbone <code>facebook/vjepa2-vitl-fpc16-256-ssv2</code>; head fine-tune khoảng 1.38M tham số; bundle báo cáo 23 MB.</li>
      <li>Nhánh probability qua Isotonic Regression rồi ánh xạ sang 1/TTC; nhánh regression dùng feature 768 chiều; output giới hạn 10 s.</li>
      <li>Dữ liệu huấn luyện CARLA DVS: 334 trips, 27,713 frames (134 collision, 200 safe); cache feature rút thử nghiệm từ khoảng 25 phút xuống mức giây.</li>
      <li>Artifact dự kiến bàn giao: <code>badas_bundle.pt</code>, notebook one-click, CSV dự đoán và đồ thị theo trip.</li>
    </ul>
    <h3>Kết quả theo sample BTC</h3>
    <table class="tight">
      <thead><tr><th>Trip</th><th>Composite</th><th>MAE (s)</th><th>F1 TTC&lt;2</th><th>FPR</th></tr></thead>
      <tbody><tr><td>T03</td><td>54.3</td><td>0.507</td><td>0.558</td><td>0.081</td></tr><tr><td>T02</td><td>48.6</td><td>5.604</td><td>0.971</td><td>0.000</td></tr><tr><td>T01</td><td>48.4</td><td>0.682</td><td>0.202</td><td>0.134</td></tr><tr><td>T06</td><td>35.5</td><td>3.374</td><td>0.750</td><td>0.074</td></tr><tr><td>T04</td><td>26.1</td><td>4.661</td><td>0.779</td><td>0.051</td></tr><tr><td>T05</td><td>10.1</td><td>17.200</td><td>0.338</td><td>0.138</td></tr></tbody>
    </table>
    <div class="callout risk"><strong>Giới hạn quan trọng:</strong> cửa sổ 16 frame tạo độ trễ xấp xỉ 1 s; inference T4 đang 165–168 s cho trip 30 s (~225 windows). Vì vậy C1 hiện là bằng chứng offline, chưa phải pipeline real-time.</div>
    """))

    sections.append(page(5, "04 · Kết quả đã hoàn thành", "C2: pipeline causal vượt tốc độ video thời gian thực", """
    <div class="grid-4">
      <div class="metric"><div class="value">81.1</div><div class="label">Composite / 6 samples<br>3,600 frames</div></div>
      <div class="metric"><div class="value">27.65</div><div class="label">FPS trên NVIDIA T4<br>input 640×384</div></div>
      <div class="metric"><div class="value">34.71</div><div class="label">Median latency, ms<br>p95: 43.19 ms</div></div>
      <div class="metric"><div class="value">75.6</div><div class="label">Peak VRAM, MiB<br>theo báo cáo C2</div></div>
    </div>
    <h3>Thiết kế model và flow kỹ thuật</h3>
    <div class="flow">
      <div class="node"><strong>One-pass CNN</strong>MobileNetV3-Large<br>global/face/eyes/mouth</div><div class="arrow">→</div>
      <div class="node"><strong>Dual ocular LSTM</strong>clean phase + robust<br>microsleep gate</div><div class="arrow">→</div>
      <div class="node"><strong>4-state LSTM</strong>causal 200 frames<br>alert/drowsy/yawn/distract</div><div class="arrow">→</div>
      <div class="node"><strong>Exclusive state</strong>5 classes<br>reset mỗi trip</div>
    </div>
    <ul>
      <li>Microsleep chỉ bật sau chuỗi mắt nhắm đủ 2 s và đủ confidence; nhánh 4-state dùng masked CE để safety gate quyết định microsleep.</li>
      <li>Fatigue episode tích lũy causal 15 s; không nhìn future frame; state được reset giữa các trip để chống rò rỉ.</li>
      <li>DMD: 364,599 frames, 80 sessions, 14 subjects ở 20 FPS; weak-label có kiểm soát, không coi khoảng trống annotation là negative.</li>
      <li>Validation theo subject-safe nested LOSO; fold hybrid đầu tiên đã audit, phạm vi cross-subject đang mở rộng.</li>
    </ul>
    <h3>Kết quả theo sample BTC</h3>
    <table class="tight">
      <thead><tr><th>Trip</th><th>Composite</th><th>Accuracy</th><th>Macro F1</th><th>Nhận xét</th></tr></thead>
      <tbody><tr><td>T01</td><td>99.5</td><td>0.995</td><td>0.995</td><td>Rất ổn định</td></tr><tr><td>T02</td><td>73.1</td><td>0.663</td><td>0.798</td><td>Còn nhầm trạng thái</td></tr><tr><td>T03</td><td>99.6</td><td>0.995</td><td>0.997</td><td>Rất ổn định</td></tr><tr><td>T04</td><td>80.1</td><td>0.805</td><td>0.797</td><td>Đạt mức khá</td></tr><tr><td>T05</td><td>94.8</td><td>0.932</td><td>0.965</td><td>Ổn định</td></tr><tr><td>T06</td><td>39.4</td><td>0.428</td><td>0.359</td><td>Case khó cần phân tích</td></tr></tbody>
    </table>
    <div class="callout warning"><strong>Rủi ro model:</strong> domain shift ở drowsy, tín hiệu chồng lấn T06, mất cân bằng microsleep, góc mặt nghiêng gây false closure. Mitigation: visibility gate, hard-negative mining, per-subject audit và checkpoint/schema fingerprint.</div>
    """))

    sections.append(page(6, "04 · Kết quả đã hoàn thành", "CarSky product slice: đã deploy, có signal live và màn hình AAOS", """
    <div class="grid-4">
      <div class="metric"><div class="value">RUNNING</div><div class="label">CarSky deployment<br>namespace room-nckuqalz</div></div>
      <div class="metric"><div class="value">11/11</div><div class="label">VSS signals có mặt<br>verify không thiếu path</div></div>
      <div class="metric"><div class="value">60/60</div><div class="label">observer updates<br>Script Node 20 Hz probe</div></div>
      <div class="metric"><div class="value">90</div><div class="label">automated tests pass<br>local integration repo</div></div>
    </div>
    <h3>Những gì chạy được ngay</h3>
    <table class="tight">
      <thead><tr><th>Hạng mục</th><th>Trạng thái</th><th>Bằng chứng định danh</th></tr></thead>
      <tbody>
        <tr><td>Blueprint <strong>SafeLoop Mock Integration Lab</strong></td><td><span class="badge verified">VALID</span></td><td><code>f0975e76-6c6b-40b5-9441-8a2dd0edcd5d</code></td></tr>
        <tr><td>Deployment</td><td><span class="badge verified">RUNNING</span></td><td><code>e691c66a-8837-4512-9133-6e4f63c88271</code></td></tr>
        <tr><td>Mock C1/C2 Fusion → Central Broker</td><td><span class="badge verified">LIVE</span></td><td>11 VSS paths; TTC, distance, speed và acceleration thay đổi.</td></tr>
        <tr><td>T01 sample GT replay</td><td><span class="badge verified">REPEATABLE</span></td><td>600 source frames @20 Hz; dashboard compact 151 samples @5 Hz.</td></tr>
        <tr><td>AAOS SafeLoop Screen</td><td><span class="badge verified">VISIBLE</span></td><td>1920×1080 WebView: TTC, DMS, contextual risk, C3 score.</td></tr>
      </tbody>
    </table>
    <figure class="shot"><img src="assets/safeloop-t01-dashboard.png" alt="SafeLoop T01 dashboard"><figcaption>Hình 1 — Screen T01-SAMPLE hiển thị C1, C2, fusion và C3. Đây là GT replay minh bạch, không phải output model live.</figcaption></figure>
    <div class="grid-2">
      <div class="callout success"><strong>Probe data-plane:</strong> Script Node timer 50 ms đã tạo 60/60 update cho ba signal ego; phù hợp hơn REST replay (~3.3 message/s trong probe 5 frame).</div>
      <div class="callout warning"><strong>Giới hạn môi trường:</strong> Conduit service chưa cấu hình nên REST ADB trả 502. Đội dùng ADB Widget của CarSky để kiểm tra AAOS; không coi đây là lỗi product.</div>
    </div>
    """))

    sections.append(page(7, "05 · Demo tính năng cốt lõi", "Kịch bản 8 phút: claim nào cũng có evidence", """
    <div class="timeline">
      <div class="timeline-row"><strong>0:00–0:45</strong><div><strong>Problem &amp; outcome.</strong> Nếu chỉ có TTC hoặc DMS riêng lẻ, hệ thống chưa biết tình huống nào cần ưu tiên. SafeLoop hợp nhất hai context.</div><span class="badge planned">SLIDE 1</span></div>
      <div class="timeline-row"><strong>0:45–2:15</strong><div><strong>C1 evidence.</strong> Mở evaluation của 6 sample; chỉ ra Composite 53.2, MAE Critical 1.780 s và case T05.</div><span class="badge provided">OFFLINE</span></div>
      <div class="timeline-row"><strong>2:15–3:30</strong><div><strong>C2 evidence.</strong> Chạy/chiếu one-click evaluation; chứng minh 81.1, 27.65 FPS và state reset.</div><span class="badge verified">MODEL</span></div>
      <div class="timeline-row"><strong>3:30–4:30</strong><div><strong>CarSky runtime.</strong> Mở deployment RUNNING và Signal Watch; quan sát Speed, acceleration, TTC, distance, DMS thay đổi.</div><span class="badge verified">LIVE</span></div>
      <div class="timeline-row"><strong>4:30–6:30</strong><div><strong>T01 core story.</strong> Phát GT replay: driver distracted 0–15 s; pedestrian jaywalk ở 15 s; min TTC ≈1.03 s; cảnh báo chuyển CRITICAL và C3 trừ điểm.</div><span class="badge progress">GT REPLAY</span></div>
      <div class="timeline-row"><strong>6:30–7:30</strong><div><strong>Failure/limitation.</strong> Nói rõ C1 chưa real-time, model chưa nối platform; chỉ ra fallback khi path lỗi hoặc payload null.</div><span class="badge blocked">HONEST</span></div>
      <div class="timeline-row"><strong>7:30–8:00</strong><div><strong>Close.</strong> Chốt product value, workstream còn lại và evidence package.</div><span class="badge planned">Q&amp;A</span></div>
    </div>
    <h3>Runbook demo T01 trên CarSky</h3>
    <ol>
      <li>Nạp môi trường: <code>source .carsky.env</code>.</li>
      <li>Kiểm tra deployment: <code>python3 tools/carsky_mock_ctl.py status</code>; yêu cầu <strong>RUNNING</strong>.</li>
      <li>Kiểm tra tín hiệu: <code>python3 tools/carsky_mock_ctl.py verify</code>; yêu cầu <code>mock_live=true</code>, <code>missing_paths=[]</code>.</li>
      <li>Mở Screen widget / Android Automotive; sinh payload bằng <code>python3 tools/carsky_screen_payload.py --html carsky/screen/safeloop_t01_dashboard.html</code>.</li>
      <li>Nếu cần tạo lại từ sample: <code>python3 tools/build_t01_demo.py</code>, sau đó phát Lua replay trên Script Node.</li>
    </ol>
    <div class="callout"><strong>Expected proof:</strong> deployment identity, signal values đổi qua ít nhất ba sample, screen chuyển trạng thái, score/penalty tích lũy, và log/manifest khớp artifact.</div>
    <div class="callout risk"><strong>Fallback có kiểm soát:</strong> nếu Screen/ADB lỗi, tiếp tục demo Signal Watch + dashboard local cùng trace T01 và công khai lý do. Không đổi sang video “giả live”.</div>
    """))

    sections.append(page(8, "06 · KPI & số liệu ban đầu", "Dashboard KPI: số đã đo, nguồn và mức tin cậy", """
    <table>
      <thead><tr><th>KPI</th><th>Kết quả</th><th>Nguồn / điều kiện</th><th>Mức bằng chứng</th></tr></thead>
      <tbody>
        <tr><td>C1 Composite</td><td><strong>53.2</strong> (6 samples)</td><td>Báo cáo C1; so với default 37.2.</td><td><span class="badge provided">TEAM REPORT</span></td></tr>
        <tr><td>C1 MAE Critical</td><td><strong>1.780 s</strong></td><td>Giảm 66.6% từ 5.338 s.</td><td><span class="badge provided">TEAM REPORT</span></td></tr>
        <tr><td>C1 runtime</td><td><strong>165–168 s / trip 30 s</strong></td><td>NVIDIA T4, khoảng 225 windows.</td><td><span class="badge provided">NOT LIVE</span></td></tr>
        <tr><td>C2 Composite</td><td><strong>81.1</strong> (3,600 frames)</td><td>6/6 sample BTC.</td><td><span class="badge provided">TEAM REPORT</span></td></tr>
        <tr><td>C2 performance</td><td><strong>27.65 FPS</strong>; p95 43.19 ms</td><td>NVIDIA T4, 640×384.</td><td><span class="badge provided">TEAM REPORT</span></td></tr>
        <tr><td>CarSky Script Node</td><td><strong>60/60 observer updates</strong></td><td>20 Hz probe, 3 ego signals.</td><td><span class="badge verified">RAW JSON</span></td></tr>
        <tr><td>Mock fusion signal coverage</td><td><strong>11/11 paths</strong></td><td>Deployment RUNNING; no missing path.</td><td><span class="badge verified">RAW JSON</span></td></tr>
        <tr><td>Integration test suite</td><td><strong>90 passed / 5.26 s</strong></td><td>Local repo snapshot 04/08/2026.</td><td><span class="badge verified">REPRODUCED</span></td></tr>
        <tr><td>T01 scenario</td><td>min TTC <strong>≈1.03 s</strong>; 30 s</td><td>Ground-truth sample replay, 600 frames.</td><td><span class="badge verified">DETERMINISTIC</span></td></tr>
      </tbody>
    </table>
    <h3>Acceptance gates trước Code Freeze</h3>
    <table class="tight">
      <thead><tr><th>Gate</th><th>Ngưỡng đội đặt ra</th><th>Hiện trạng</th></tr></thead>
      <tbody>
        <tr><td>G1 · Reproducibility</td><td>Clean run tạo đúng CSV/metrics/manifest trên 6 samples.</td><td>Có report; cần gom code, checkpoint và hash.</td></tr>
        <tr><td>G2 · E2E product</td><td>Ít nhất một sample chạy model thật → fusion → CarSky → Screen.</td><td>Chưa đạt; hiện mock/GT.</td></tr>
        <tr><td>G3 · Latency</td><td>C2 ≥20 FPS; C1 có profile và mode demo phù hợp.</td><td>C2 đạt; C1 chưa đạt live.</td></tr>
        <tr><td>G4 · Robustness</td><td>Reset trip, missing/null input, broker restart có expected behavior.</td><td>Một phần đã test; cần fault matrix E2E.</td></tr>
        <tr><td>G5 · Evidence identity</td><td>Commit + config + model hash + run ID + video khớp nhau.</td><td>Platform có; model artifacts cần nhập repo.</td></tr>
      </tbody>
    </table>
    """))

    sections.append(page(9, "07 · Khó khăn, rủi ro & hỗ trợ", "Đường găng không nằm ở UI mà ở integration contract", """
    <table>
      <thead><tr><th>Rủi ro</th><th>Tác động</th><th>Mitigation / quyết định</th><th>Hỗ trợ mong muốn</th></tr></thead>
      <tbody>
        <tr><td><strong>C1 chưa real-time</strong><br>window lag + T4 runtime cao</td><td>Không thể claim live collision model.</td><td>Profile, FP16/batch, giảm window/stride có kiểm soát; giữ chế độ offline replay nếu chất lượng giảm.</td><td>Mentor góp ý trade-off metric–latency và target compute trên xe.</td></tr>
        <tr><td><strong>C2 domain shift/T06</strong></td><td>Giảm Macro F1, cảnh báo sai.</td><td>Subject-safe LOSO, visibility gate, hard-negative mining, per-state confusion audit.</td><td>Xác nhận protocol/label semantics cho microsleep–drowsy.</td></tr>
        <tr><td><strong>Model artifacts phân tán</strong></td><td>Không tái lập được claim từ integration repo.</td><td>Artifact contract, checksum, lockfile, one-click command, shared evidence index.</td><td>Không cần hạ tầng mới; các workstream kỹ thuật cần bàn giao đúng gate.</td></tr>
        <tr><td><strong>Broker → Screen chưa live</strong></td><td>Screen đẹp nhưng chưa phản ánh signal runtime.</td><td>Xây bridge subscriber nhỏ; timestamp/sequence/health indicator; stale-data fallback.</td><td>CarSky guidance cho kênh dữ liệu chính thức đến AAOS app.</td></tr>
        <tr><td><strong>Conduit chưa cấu hình</strong></td><td>REST ADB 502; automation Screen hạn chế.</td><td>Dùng ADB Widget để demo; lưu payload/runbook; không phụ thuộc Conduit.</td><td>BTC xác nhận đây là giới hạn tenant hay cấu hình có thể bật.</td></tr>
        <tr><td><strong>Safety / privacy</strong></td><td>Over-warning, video cabin nhạy cảm.</td><td>Fail-silent, advisory only, local feature extraction, không lưu raw video mặc định, audit trail.</td><td>Mentor review HMI escalation và data-retention.</td></tr>
      </tbody>
    </table>
    <h3>Failure paths đã khảo sát trên CarSky</h3>
    <div class="grid-2">
      <div class="card red"><h3>Payload / schema</h3><ul><li>Unknown signal: HTTP 400.</li><li>Float null bị từ chối; batch có thể không atomic.</li><li>Cyclic route không được hỗ trợ.</li></ul></div>
      <div class="card amber"><h3>Runtime / API</h3><ul><li>Runtime route có sai khác theo API surface.</li><li>Conduit missing: HTTP 502.</li><li>Credential format và method mismatch đã được xử lý trong CLI.</li></ul></div>
    </div>
    <div class="callout warning"><strong>Nguyên tắc an toàn:</strong> mọi tín hiệu stale/missing/schema mismatch đều phải hạ mức tin cậy hoặc tắt can thiệp; không suy diễn giá trị “an toàn” từ dữ liệu hỏng.</div>
    """))

    sections.append(page(10, "08 · Kế hoạch công việc còn lại", "Năm gate theo dependency, không khóa vào mốc ngày sai", """
    <div class="timeline">
      <div class="timeline-row"><strong>Gate 1</strong><div><strong>Artifact Freeze C1/C2.</strong> Đưa code, checkpoint/bundle, config, dependency lock và lệnh reproduce vào cùng submission snapshot; ghi SHA-256.</div><span class="badge progress">ƯU TIÊN 1</span></div>
      <div class="timeline-row"><strong>Gate 2</strong><div><strong>Inference Contract.</strong> Chuẩn hóa timestamp/frame_id, confidence, state reset, missing-data semantics và output schema dùng chung.</div><span class="badge progress">ƯU TIÊN 1</span></div>
      <div class="timeline-row"><strong>Gate 3</strong><div><strong>Model → CarSky.</strong> Tích hợp C2 trước vì đã đạt real-time; với C1 khóa một mode chính thức sau profile (live tối ưu hoặc offline/replay được công bố).</div><span class="badge planned">ĐƯỜNG GĂNG</span></div>
      <div class="timeline-row"><strong>Gate 4</strong><div><strong>Broker → Screen live.</strong> Thay trace nhúng bằng subscriber bridge; hiển thị health, age và source mode MODEL/MOCK/GT.</div><span class="badge planned">ĐƯỜNG GĂNG</span></div>
      <div class="timeline-row"><strong>Gate 5</strong><div><strong>Evidence Freeze.</strong> Chạy 6 sample, fault matrix, latency, video ≤10 phút; khóa Claim–Evidence Map và dry-run Q&amp;A.</div><span class="badge planned">CHỐT NỘP</span></div>
    </div>
    <h3>Definition of Done</h3>
    <table class="tight">
      <thead><tr><th>Workstream</th><th>Hoàn thành khi</th><th>Không chấp nhận</th></tr></thead>
      <tbody>
        <tr><td>C1</td><td>6 sample reproducible; report khớp bundle hash; latency profile có quyết định mode.</td><td>Chỉ có số trong slide/DOCX.</td></tr>
        <tr><td>C2</td><td>6 sample + causal/reset tests; checkpoint/schema fingerprint; adapter phát output thật.</td><td>Đánh giá trộn subject hoặc state leak.</td></tr>
        <tr><td>C3</td><td>Policy versioned; reason/action deterministic; test edge cases và fairness.</td><td>Score “magic number” không giải thích.</td></tr>
        <tr><td>CarSky</td><td>Model output xuất hiện trên Central Broker và AAOS; restart/replay lặp lại được.</td><td>Screen độc lập với signal runtime.</td></tr>
        <tr><td>Submission</td><td>Commit/artifact/config/video/claim map cùng một identity.</td><td>Demo và tài liệu dùng phiên bản khác nhau.</td></tr>
      </tbody>
    </table>
    <div class="callout success"><strong>Thứ tự ưu tiên:</strong> khóa reproducibility và model-to-platform trước; polishing UI, fleet map và coaching nâng cao chỉ thực hiện sau Gate 4.</div>
    """))

    sections.append(page(11, "09 · Phân công nhiệm vụ trong đội", "Ownership theo evidence chain thực tế", """
    <table>
      <thead><tr><th>Owner / workstream</th><th>Phạm vi thực tế đợt 2</th><th>Deliverable chịu trách nhiệm</th><th>Gate kế tiếp</th></tr></thead>
      <tbody>
        <tr><td><strong>Nguyễn Thanh Sơn</strong><br>Product &amp; Integration Lead</td><td>Product framing; telemetry contract; CarSky; C3 mock; T01 replay; AAOS Screen; báo cáo/demo.</td><td>Blueprint/deployment, CLI/runbook, Screen, test/evidence package, unified report.</td><td>Model adapters, broker-screen bridge, demo freeze.</td></tr>
        <tr><td><strong>AnhNH2225</strong><br>C1 technical</td><td>V-JEPA2 fine-tune/head; CARLA training; calibrator và sweep.</td><td>C1 source/config, <code>badas_bundle.pt</code>, evaluation CSV/plots.</td><td>Artifact handoff, latency profile/optimization.</td></tr>
        <tr><td><strong>TuHV12</strong><br>C1 technical</td><td>CARLA/calibrator/sweep; bundle, one-click notebook và plots.</td><td>Reproducible C1 notebook, bundle manifest, per-trip evidence.</td><td>Clean-run verification và CarSky adapter support.</td></tr>
        <tr><td><strong>Nhóm C2 · 2 thành viên</strong><br>C2 technical</td><td>MobileNetV3-Large, dual ocular LSTM, causal four-state LSTM, validation và optimization.</td><td>C2 checkpoint/config, schema candidate 7, evaluation CSV, latency/confusion evidence.</td><td>Khóa owner cụ thể trong bản nộp; bàn giao artifact và inference adapter.</td></tr>
      </tbody>
    </table>
    <div class="callout warning"><strong>Điểm cần trưởng nhóm khóa trước khi gửi:</strong> báo cáo C2 không chứa tên hai tác giả trong metadata/nội dung. Bản này không tự suy đoán tên để tránh sai ownership; thay dòng “Nhóm C2 · 2 thành viên” bằng tên chính thức khi xác nhận.</div>
    <h3>RACI rút gọn</h3>
    <table class="tight">
      <thead><tr><th>Deliverable</th><th>Responsible</th><th>Accountable</th><th>Consulted / reviewer</th></tr></thead>
      <tbody><tr><td>C1 model evidence</td><td>AnhNH2225, TuHV12</td><td>C1 workstream</td><td>Sơn · integration contract</td></tr><tr><td>C2 model evidence</td><td>2 thành viên C2</td><td>C2 workstream</td><td>Sơn · integration contract</td></tr><tr><td>C3 + CarSky + Screen</td><td>Sơn</td><td>Sơn</td><td>C1/C2 owners · output schema</td></tr><tr><td>Final demo/report</td><td>Sơn + toàn đội</td><td>Sơn</td><td>Mỗi owner ký claim của mình</td></tr></tbody>
    </table>
    <div class="callout"><strong>Working agreement:</strong> mỗi claim kỹ thuật phải có owner, command tái lập, raw output và artifact hash; trưởng nhóm chỉ hợp nhất, không thay owner xác nhận số liệu model.</div>
    """))

    sections.append(page("A", "Phụ lục A · Claim–Evidence Map", "Mỗi claim trỏ tới một bằng chứng kiểm tra được", """
    <table class="tight">
      <thead><tr><th>ID</th><th>Claim</th><th>Evidence</th><th>Trạng thái / giới hạn</th></tr></thead>
      <tbody>
        <tr><td>C1-01</td><td>C1 Composite 53.2; MAE Critical 1.780 s.</td><td><code>docs/dot2_team/BAO_CAO_TIEN_DO_C2_CHALLENGE1.docx</code><br>SHA-256: <code>f51b117…d50f</code></td><td><span class="badge provided">REPORTED</span><br>Cần import raw run + bundle.</td></tr>
        <tr><td>C1-02</td><td>C1 chưa real-time trên T4.</td><td>Cùng báo cáo: 165–168 s / trip 30 s.</td><td><span class="badge provided">LIMIT</span></td></tr>
        <tr><td>C2-01</td><td>C2 Composite 81.1; 27.65 FPS.</td><td><code>docs/dot2_team/BAO_CAO_TIEN_DO_C2_CHALLENGE2 1.docx</code><br>SHA-256: <code>55e170f…598a</code></td><td><span class="badge provided">REPORTED</span><br>Cần import raw run + checkpoint.</td></tr>
        <tr><td>PL-01</td><td>SafeLoop deployment RUNNING.</td><td><code>tools/carsky_mock_ctl.py status</code><br>deployment <code>e691c66a…</code></td><td><span class="badge verified">LIVE CHECK</span></td></tr>
        <tr><td>PL-02</td><td>11/11 signal mock có mặt; 5 path thay đổi trong 3 samples.</td><td><code>reports/evidence/carsky-mock-c1-c2-runtime-20260804.json</code><br>SHA-256: <code>5849d3d…faf94</code></td><td><span class="badge verified">RAW JSON</span></td></tr>
        <tr><td>PL-03</td><td>Script Node 20 Hz tạo 60/60 observer updates.</td><td><code>reports/evidence/carsky-script-node-20hz-probe-20260803.json</code><br>SHA-256: <code>ceab528…775e</code></td><td><span class="badge verified">RAW JSON</span></td></tr>
        <tr><td>DEMO-01</td><td>T01 GT replay và Screen tái lập được.</td><td><code>tools/build_t01_demo.py</code>, <code>carsky/scripts/safeloop_t01_gt_replay.lua</code>, <code>carsky/screen/safeloop_t01_dashboard.html</code></td><td><span class="badge progress">GT REPLAY</span><br>Không phải model inference.</td></tr>
        <tr><td>QA-01</td><td>Integration repo có 90 automated tests pass.</td><td><code>python3 -m pytest -q</code> → <code>90 passed in 5.26s</code></td><td><span class="badge verified">REPRODUCED</span></td></tr>
        <tr><td>FAIL-01</td><td>Đã khảo sát lỗi schema/runtime/Conduit.</td><td><code>reports/evidence/carsky-failure-paths-20260803.json</code></td><td><span class="badge verified">RAW JSON</span></td></tr>
      </tbody>
    </table>
    <div class="callout warning"><strong>Artifact identity hiện tại:</strong> branch <code>feat/task-1.2-tripkit</code>, base commit <code>c8506f4</code>; worktree chứa thay đổi chưa commit. Trước khi nộp phải tạo commit/tag sạch và sinh lại manifest.</div>
    <h3>Quy ước nhãn</h3>
    <p><span class="badge verified">VERIFIED</span> chạy/đọc được trong integration repo hoặc CarSky; <span class="badge provided">REPORTED</span> do workstream C1/C2 cung cấp, chưa tái chạy trong repo này; <span class="badge progress">MOCK/GT</span> test double/replay công khai; <span class="badge planned">PLANNED</span> chưa hoàn tất.</p>
    """, "appendix-a"))

    sections.append(page("B", "Phụ lục B · Barem coverage", "Kế hoạch bằng chứng theo 100 điểm vòng 2", """
    <table class="tight">
      <thead><tr><th>Nhóm tiêu chí</th><th>Điểm</th><th>Bằng chứng hiện có</th><th>Việc phải khóa để đạt mức cao</th></tr></thead>
      <tbody>
        <tr><td>Demo E2E / core</td><td class="score-col">25</td><td>C1/C2 evidence + CarSky mock/GT + AAOS Screen.</td><td>Ít nhất một sample model thật xuyên suốt; demo ≤10 phút.</td></tr>
        <tr><td>Technical quality / evidence</td><td class="score-col">20</td><td>Model architecture, per-trip metrics, raw JSON, tests, failure paths.</td><td>Gom raw model run/checkpoint/config/hash; clean reproduce.</td></tr>
        <tr><td>Giá trị gia tăng do đội</td><td class="score-col">25</td><td>Fusion policy, C3, VSS contract, CarSky controller, T01 builder, Screen.</td><td>Đo ablation C1-only/C2-only/fusion và trip-score usefulness.</td></tr>
        <tr><td>Platform / ecosystem</td><td class="score-col">15</td><td>Blueprint hợp lệ, deployment RUNNING, broker signals, AAOS.</td><td>Model adapter + broker-screen bridge; restart/fault proof.</td></tr>
        <tr><td>User / customer / deployability</td><td class="score-col">10</td><td>Fleet/driver use case, advisory UI, privacy stance.</td><td>User test ngắn, HMI rationale, compute/cost/deployment profile.</td></tr>
        <tr><td>Trình bày / Q&amp;A</td><td class="score-col">5</td><td>8-minute runbook, limitation disclosure, claim map.</td><td>Dry-run, backup path, owner Q&amp;A cards.</td></tr>
      </tbody>
    </table>
    <div class="callout success"><strong>Chiến lược chấm điểm:</strong> không che khoảng trống bằng UI. Mỗi phút demo phải nối được Product → Technical → Evidence → CarSky → User value.</div>
    <h3>Gói deliverable đề xuất</h3>
    <ul>
      <li>PDF/HTML báo cáo này và slide demo ngắn.</li>
      <li>Video demo ≤10 phút, có model evidence + CarSky runtime + AAOS Screen.</li>
      <li>Repo/tag sạch, README one-click, dependency lock, Product &amp; Integration Card.</li>
      <li>Model artifacts/config/hash, raw results 6 sample, platform evidence JSON.</li>
      <li>Claim–Evidence Map và bảng limitation/failure-path.</li>
    </ul>
    """, "appendix-a"))

    sections.append(page("C", "Phụ lục C · Minh bạch model và dữ liệu", "Reproducibility, safety và những điều chưa chứng minh", """
    <table>
      <thead><tr><th>Thành phần</th><th>Dữ liệu / model</th><th>Validation</th><th>Giới hạn</th></tr></thead>
      <tbody>
        <tr><td>C1 TTC</td><td>CARLA DVS; V-JEPA2 ViT-L frozen + head lai.</td><td>6 sample BTC; metrics theo trip; critical-region metrics.</td><td>Lag ~1 s; T05 yếu; T4 runtime chưa real-time.</td></tr>
        <tr><td>C2 DMS</td><td>DMD 364,599 frames / 80 sessions / 14 subjects; MobileNetV3 + LSTM.</td><td>Subject-safe nested LOSO đang mở rộng; 6 sample BTC.</td><td>Domain shift; T06; class imbalance; profile-view occlusion.</td></tr>
        <tr><td>C3 Fusion</td><td>Rule-based prototype; TTC + driver risks + ego context.</td><td>Unit tests, mock runtime, T01 deterministic replay.</td><td>Chưa calibrate từ model output / user study; chưa safety-certified.</td></tr>
        <tr><td>CarSky Screen</td><td>Mock trace hoặc T01 GT trace nhúng; AAOS WebView.</td><td>Quan sát ở 1920×1080 và lệnh payload tái lập.</td><td>Chưa broker subscriber live; WebView là prototype HMI.</td></tr>
      </tbody>
    </table>
    <h3>Privacy &amp; safety by design</h3>
    <div class="grid-2">
      <div class="card green"><h3>Data minimization</h3><p>Ưu tiên xử lý cabin cục bộ và chỉ phát feature/state cần thiết. Không lưu raw cabin video mặc định; retention phải có mục đích và consent.</p></div>
      <div class="card blue"><h3>Advisory, fail-silent</h3><p>Prototype chỉ khuyến nghị/cảnh báo, không điều khiển xe. Input stale/missing hoặc model mismatch sẽ hạ confidence và vô hiệu escalation.</p></div>
    </div>
    <h3>Tuyên bố sử dụng AI hỗ trợ</h3>
    <p>AI coding assistant được dùng để hỗ trợ thiết kế integration, tạo test, phân tích API, dựng mock/GT replay, Screen và tổng hợp báo cáo. Các số C1/C2 do owner workstream cung cấp; các bằng chứng CarSky/test được lưu dưới dạng command/output. AI không được dùng làm oracle cho ground truth hay tự xác nhận metric model.</p>
    <div class="callout risk"><strong>Chưa chứng minh:</strong> hiệu quả safety ngoài 6 sample, generalization ngoài domain, độ tin cậy production, tác động giảm tai nạn, fairness giữa nhóm người lái và khả năng can thiệp actuator. Đây là mục tiêu kiểm thử tiếp theo, không phải claim hiện tại.</div>
    <div class="callout success"><strong>Kết luận giữa kỳ:</strong> đội đã có hai nền tảng AI kỹ thuật đáng kể và một product slice CarSky nhìn thấy được. Đường găng rõ ràng là đóng gói artifact và thay mock/GT bằng model adapters trong cùng evidence chain.</div>
    """))

    # The BTC-facing v2 requested by the team lead stops after section 08.
    # Index 0 is the cover; indexes 1..10 contain sections 01..08 because
    # section 04 spans three pages.
    if compact_v2:
        sections = sections[:11]
    body = "\n".join(sections)
    return simplify_v2_language(body) if compact_v2 else body


def build_html(style_source: Path, compact_v2: bool = False) -> str:
    source = style_source.read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", source, flags=re.S)
    if not match:
        raise ValueError(f"No stylesheet found in {style_source}")
    css = match.group(1)
    if ".architecture {" not in css:
        css += """
    .architecture { border: 1px solid var(--line); border-radius: 3mm; padding: 2mm; background: #fbfdff; }
    .architecture svg { display: block; width: 100%; height: auto; font-family: Inter, 'Segoe UI', Arial, sans-serif; font-size: 18px; fill: var(--ink); }
    .legend { margin-top: 2mm; }
    .shot { margin: 2mm 0 2.5mm; }
    .shot img { display: block; width: 100%; max-height: 86mm; object-fit: cover; object-position: center top; border: 1px solid var(--line); border-radius: 2.5mm; }
    .shot figcaption { color: var(--muted); font-size: 7.8pt; margin-top: 1mm; }
    @media print { html, body { background: white; } .page { margin: 0; } }
    """
    title = "Báo cáo tiến độ đợt 2 — 5 Chàng Lính Ngự Lâm — SafeLoop"
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
{build_body(compact_v2=compact_v2)}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-source", type=Path, default=STYLE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compact-v2",
        action="store_true",
        help="Stop after section 08; omit section 09 and appendices A/B/C",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(args.style_source.resolve(), compact_v2=args.compact_v2),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
