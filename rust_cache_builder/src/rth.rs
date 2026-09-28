/// RTH (Regular Trading Hours) filtering and timestamp utilities.
///
/// Mirrors Python's timestamp functions in rebuild_snapshot_caches.py.
///
/// RTH for ES futures (used in Python code):
///   9:30 AM - 4:00 PM ET
///   = 13:30 - 20:00 UTC (EDT, UTC-4, Mar-Nov)
///   = 14:30 - 21:00 UTC (EST, UTC-5, Nov-Mar)
///
/// DST_END_2025: Nov 2, 2025 06:00 UTC

/// DST end 2025: Nov 2, 2025 06:00 UTC in nanoseconds
const DST_END_2025_NS: u64 = 1_762_056_000_000_000_000;

/// RTH start and end in minutes from midnight ET
const RTH_START_MINUTES: u32 = 9 * 60 + 30;   // 9:30 AM ET
const RTH_END_MINUTES: u32 = 16 * 60;           // 4:00 PM ET

/// Prime hours start and end in minutes from midnight ET
const PRIME_START_MINUTES: u32 = 10 * 60 + 30; // 10:30 AM ET
const PRIME_END_MINUTES: u32 = 14 * 60 + 30;   // 2:30 PM ET

/// Get UTC offset for Eastern Time: -4 (EDT) or -5 (EST).
fn et_offset_hours(timestamp_ns: u64) -> i64 {
    if timestamp_ns < DST_END_2025_NS { -4 } else { -5 }
}

/// Check if a nanosecond UTC timestamp falls within RTH (9:30-16:00 ET, weekdays only).
pub fn is_within_rth(timestamp_ns: u64) -> bool {
    if timestamp_ns == 0 { return false; }

    let ts_sec = (timestamp_ns / 1_000_000_000) as i64;
    let et_offset = et_offset_hours(timestamp_ns);
    let et_sec = ts_sec + et_offset * 3600;

    // Check weekday. Jan 1, 1970 was a Thursday (day 4, 0=Sun).
    // (days_since_epoch + 4) % 7: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
    let days_since_epoch = et_sec.div_euclid(86400);
    let day_of_week = ((days_since_epoch + 4).rem_euclid(7)) as u32;
    // 0=Sunday, 6=Saturday → skip weekends
    if day_of_week == 0 || day_of_week == 6 {
        return false;
    }

    let secs_in_day = et_sec.rem_euclid(86400) as u32;
    let hour = secs_in_day / 3600;
    let minute = (secs_in_day % 3600) / 60;
    let time_minutes = hour * 60 + minute;

    time_minutes >= RTH_START_MINUTES && time_minutes < RTH_END_MINUTES
}

/// Check if a nanosecond UTC timestamp falls within prime trading hours (10:30 AM - 2:30 PM ET,
/// weekdays only). Must be called only for timestamps already confirmed within RTH.
pub fn is_within_prime_hours(timestamp_ns: u64) -> bool {
    if timestamp_ns == 0 { return false; }

    let ts_sec = (timestamp_ns / 1_000_000_000) as i64;
    let et_offset = et_offset_hours(timestamp_ns);
    let et_sec = ts_sec + et_offset * 3600;

    // Weekday check (same as is_within_rth)
    let days_since_epoch = et_sec.div_euclid(86400);
    let day_of_week = ((days_since_epoch + 4).rem_euclid(7)) as u32;
    if day_of_week == 0 || day_of_week == 6 {
        return false;
    }

    let secs_in_day = et_sec.rem_euclid(86400) as u32;
    let hour = secs_in_day / 3600;
    let minute = (secs_in_day % 3600) / 60;
    let time_minutes = hour * 60 + minute;

    time_minutes >= PRIME_START_MINUTES && time_minutes < PRIME_END_MINUTES
}

/// Convert nanosecond UTC timestamp to Eastern Time calendar date string "YYYY-MM-DD".
/// Only reliable when called during RTH (no overnight boundary ambiguity).
pub fn ts_to_et_date(timestamp_ns: u64) -> String {
    let ts_sec = (timestamp_ns / 1_000_000_000) as i64;
    let et_offset = et_offset_hours(timestamp_ns);
    let et_sec = ts_sec + et_offset * 3600;

    // Convert to date using integer arithmetic (no chrono needed)
    // Days since Unix epoch
    let days = et_sec.div_euclid(86400);
    let (year, month, day) = days_to_ymd(days as i32);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Check if a date string "YYYY-MM-DD" is a weekday (Mon-Fri).
pub fn is_weekday(date_str: &str) -> bool {
    // Parse the date and compute day of week
    let parts: Vec<u32> = date_str.splitn(3, '-')
        .filter_map(|s| s.parse().ok())
        .collect();
    if parts.len() != 3 { return false; }
    let (y, m, d) = (parts[0], parts[1], parts[2]);
    let dow = day_of_week(y, m, d);
    // Tomohiko Sakamoto: 0=Sunday, 1=Monday, ..., 5=Friday, 6=Saturday
    // Weekdays are 1 (Mon) through 5 (Fri) inclusive.
    dow >= 1 && dow <= 5
}

/// Days since Unix epoch (1970-01-01) to (year, month, day).
/// Uses the proleptic Gregorian calendar algorithm.
fn days_to_ymd(z: i32) -> (i32, u32, u32) {
    // From https://howardhinnant.github.io/date_algorithms.html
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Day of week: 0=Mon, 1=Tue, ..., 6=Sun. Uses Tomohiko Sakamoto's algorithm.
fn day_of_week(y: u32, m: u32, d: u32) -> u32 {
    let t: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    (y + y / 4 - y / 100 + y / 400 + t[(m - 1) as usize] + d) % 7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rth_filter() {
        // 2025-07-14 15:30 UTC = 11:30 AM EDT = within RTH (Monday)
        // 2025-07-14 00:00 UTC = 1752451200 + 15.5h = 1752451200 + 55800 = 1752507000
        let within = 1752507000_u64 * 1_000_000_000;
        assert!(is_within_rth(within), "11:30 AM EDT Monday should be RTH");

        // 2025-07-14 20:30 UTC = 16:30 PM EDT = outside RTH (after 4 PM)
        // 1752451200 + 20.5*3600 = 1752451200 + 73800 = 1752525000
        let outside = 1752525000_u64 * 1_000_000_000;
        assert!(!is_within_rth(outside), "16:30 EDT Monday should be outside RTH");

        // 2025-07-14 12:00 UTC = 8:00 AM EDT = before RTH (before 9:30 AM)
        // 1752451200 + 12*3600 = 1752451200 + 43200 = 1752494400
        let pre_rth = 1752494400_u64 * 1_000_000_000;
        assert!(!is_within_rth(pre_rth), "8:00 AM EDT Monday should be before RTH");

        // 2025-07-13 13:30 UTC = 9:30 AM EDT = Sunday — NOT RTH even though time matches
        // July 13 midnight UTC = 1752364800 + 13:30h = 1752364800 + 48600 = 1752413400
        let sunday_930 = 1752413400_u64 * 1_000_000_000;
        assert!(!is_within_rth(sunday_930), "Sunday 9:30 AM should NOT be RTH");

        // 2025-07-12 15:00 UTC = 11:00 AM EDT = Saturday — NOT RTH
        // July 12 midnight UTC = 1752278400 + 15*3600 = 1752278400 + 54000 = 1752332400
        let saturday = 1752332400_u64 * 1_000_000_000;
        assert!(!is_within_rth(saturday), "Saturday should NOT be RTH");

        // 2025-07-18 15:00 UTC = 11:00 AM EDT = Friday — IS RTH
        // July 18 midnight UTC = 1752451200 + 7*86400 = 1752451200 + 604800 = 1753056000
        // Wait: Jul 14 midnight UTC = 1752451200, Jul 18 midnight = 1752451200 + 4*86400 = 1752451200 + 345600 = 1752796800
        // Jul 18 15:00 UTC = 1752796800 + 15*3600 = 1752796800 + 54000 = 1752850800
        let friday_rth = 1752850800_u64 * 1_000_000_000;
        assert!(is_within_rth(friday_rth), "11:00 AM EDT Friday should be RTH");
    }

    #[test]
    fn test_date_conversion() {
        // 2025-07-14 15:00 UTC = 11:00 AM EDT
        let ts = 1752502800_u64 * 1_000_000_000;
        let date = ts_to_et_date(ts);
        assert_eq!(date, "2025-07-14");
    }

    #[test]
    fn test_weekday() {
        assert!(is_weekday("2025-07-14")); // Monday
        assert!(is_weekday("2025-07-15")); // Tuesday
        assert!(is_weekday("2025-07-16")); // Wednesday
        assert!(is_weekday("2025-07-17")); // Thursday
        assert!(is_weekday("2025-07-18")); // Friday — was incorrectly classified as weekend before the fix
        assert!(is_weekday("2025-09-19")); // Friday (ESU5->ESZ5 rollover day)
        assert!(!is_weekday("2025-07-12")); // Saturday
        assert!(!is_weekday("2025-07-13")); // Sunday
    }
}
