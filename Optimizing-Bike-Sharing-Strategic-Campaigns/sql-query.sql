SELECT * FROM `bigqueryprojects-385010.case_study.cyclistic` LIMIT 1000

-- Difference in the number of members and casual riders
SELECT member_casual,
       COUNT(ride_id) AS total,
       CONCAT(ROUND((COUNT(ride_id) / SUM(COUNT(ride_id)) OVER ()) * 100), '%') AS percentage
FROM `case_study.cyclistic`
GROUP BY 1

-- Total Bike Type
SELECT member_casual,
       rideable_type,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic`
GROUP BY 1, 2


-- Total Annual Member and Casual Riders by Months
-- Casual
SELECT member_casual,
       FORMAT_DATE('%B', started_at) AS month,
       COUNT(ride_id) total_casual_riders
FROM `case_study.cyclistic`
WHERE member_casual = 'casual'
GROUP BY 1, 2
ORDER BY 3 DESC
-- Member
SELECT member_casual,
       FORMAT_DATE('%B', started_at) AS month,
       COUNT(ride_id) total_casual_riders
FROM `case_study.cyclistic`
WHERE member_casual = 'member'
GROUP BY 1, 2
ORDER BY 3 DESC

-- Total Annual Members and Casual Riders by Day of The Week
-- Member
SELECT member_casual,
       day_name,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'member'
GROUP BY 1, 2
ORDER BY 1, 3 DESC
-- Casual
SELECT member_casual,
       day_name,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'casual'
GROUP BY 1, 2
ORDER BY 1, 3 DESC

-- Total Annual Members and Casual Riders by Hour
-- Member
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'member'
GROUP BY 1, 2
ORDER BY 1, 3 DESC
-- Casual
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'casual'
GROUP BY 1, 2
ORDER BY 1, 3 DESC

-- Total Annual Members and Casual Riders by Hours in Weekday
-- Member
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour_in_weekday,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'member' AND is_weekend = false
GROUP BY 1, 2
ORDER BY 1, 3 DESC
-- Casual
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour_in_weekday,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'casual' AND is_weekend = false
GROUP BY 1, 2
ORDER BY 1, 3 DESC

-- Total Annual Members and Casual Riders by Hours in weekend
-- Member
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour_in_weekend,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'member' AND is_weekend = true
GROUP BY 1, 2
ORDER BY 1, 3 DESC
-- Casual
SELECT member_casual,
       EXTRACT(HOUR FROM started_at) AS hour_in_weekend,
       COUNT(ride_id) AS total
FROM `case_study.cyclistic_cleaned`
WHERE member_casual = 'casual' AND is_weekend = true
GROUP BY 1, 2
ORDER BY 1, 3 DESC

-- Distribution of Total Bike Users Based on a Duration
SELECT member_casual,
       CASE
        WHEN duration_minutes >= 0 AND duration_minutes < 20 THEN '0-20 minutes'
        WHEN duration_minutes >= 20 AND duration_minutes < 40 THEN '20-40 minutes'
        WHEN duration_minutes >= 40 AND duration_minutes < 60 THEN '40-60 minutes'
        ELSE 'more than 60 minutes'
      END AS duration_range,
      COUNT(ride_id) AS total_users
FROM `case_study.cyclistic`
GROUP BY 1, 2
ORDER BY 1, 2











