SELECT 
  p.id AS project_id,
  p.url AS project_url,
  p.description,
  pc.latest_commit_date,
  pc.medium_term_commit_count,
  pc.medium_term_distinct_author_count,
  pc.medium_term_distinct_committer_count,
  pc.recent_commit_count,
  pc.recent_distinct_author_count,
  pc.recent_distinct_committer_count,
  prstats.latest_pull_request_history_date,
  prstats.medium_term_pull_request_count,
  prstats.recent_pull_request_count,
  p.LANGUAGE AS project_language,
  pl.language AS project_language_details_language,
  pl.bytes AS project_language_bytes,
  pl.language_percentage,
  pl.created_at AS project_language_created_at,  
  p.forked_from,
  pr.id AS pull_request_id,
  pr.pullreq_id,
  pr.intra_branch,
  prc.user_id,
  prc.comment_id,
  prc.position,
  prc.body,
  prc.commit_id,
  prc.created_at
FROM `ghtorrent-bq.ght_2018_04_01.projects` AS p
  INNER JOIN (
    -- Subquery to get projects that have recent and medium term activities, to ensure the selected projects are active and sustaining.
    SELECT
      pc.project_id,
      MAX(c.created_at) AS latest_commit_date,
      COUNT(DISTINCT author_id) AS medium_term_distinct_author_count,
      COUNT(DISTINCT committer_id) AS medium_term_distinct_committer_count,
      COUNT(commit_id) AS medium_term_commit_count,
      COUNT(DISTINCT CASE WHEN c.created_at >= '2018-01-01' THEN author_id END) AS recent_distinct_author_count,
      COUNT(DISTINCT CASE WHEN c.created_at >= '2018-01-01' THEN committer_id END) AS recent_distinct_committer_count,
      COUNT(CASE WHEN c.created_at >= '2018-01-01' THEN commit_id END) AS recent_commit_count
    FROM `ghtorrent-bq.ght_2018_04_01.project_commits` AS pc 
      INNER JOIN `ghtorrent-bq.ght_2018_04_01.commits` AS c ON c.id = pc.commit_id
    WHERE c.created_at >= '2016-01-01' -- Medium Term activity
    -- AND c.created_at < CURRENT_TIMESTAMP() -- Filter invalid commit dates
    GROUP BY pc.project_id
  ) AS pc ON pc.project_id = p.id
  INNER JOIN (
    -- Uses Pull Request, and have recent and medium term activities.
    SELECT 
      base_repo_id,
      MAX(prh.created_at) AS latest_pull_request_history_date,
      COUNT(DISTINCT pr.id) AS medium_term_pull_request_count,
      COUNT(DISTINCT CASE WHEN prh.created_at >= '2018-01-01' THEN pr.id END) AS recent_pull_request_count      
    FROM `ghtorrent-bq.ght_2018_04_01.pull_requests` AS pr
      INNER JOIN `ghtorrent-bq.ght_2018_04_01.pull_request_history` AS prh ON prh.pull_request_id = pr.id
    WHERE prh.created_at >= '2016-01-01' -- Medium Term activity
    -- Do we case if the pull request is Closed (abandoned) or Merged?
    -- The peril paper suggests the pull request may be merged from commits with a "Fixes" keyword.
    GROUP BY base_repo_id
  ) AS prstats ON prstats.base_repo_id = p.id 
  LEFT JOIN (
    -- Subquery to get projects that have Java as one of the prominent languages
    SELECT 
      pl.project_id,
      pl.language,
      pl.created_at,
      pl.bytes,
      pl.bytes / pl_latest_total_bytes.total_bytes * 100 AS language_percentage
    FROM (        
      SELECT
        pl.project_id,
        pl.created_at,
        SUM(bytes) AS total_bytes
      FROM (
        SELECT
          project_id,
          MAX(created_at) AS latest_refresh_date
        FROM `ghtorrent-bq.ght_2018_04_01.project_languages`
        GROUP BY project_id
      ) AS pl_latest
        INNER JOIN `ghtorrent-bq.ght_2018_04_01.project_languages` AS pl ON pl.project_id = pl_latest.project_id AND pl.created_at = pl_latest.latest_refresh_date
      GROUP BY pl.project_id, pl.created_at       
    ) AS pl_latest_total_bytes
      INNER JOIN `ghtorrent-bq.ght_2018_04_01.project_languages` AS pl ON pl.project_id = pl_latest_total_bytes.project_id AND pl.created_at = pl_latest_total_bytes.created_at
    WHERE LOWER(pl.language) = 'java' 
    AND pl_latest_total_bytes.total_bytes > 0
    AND pl.bytes / pl_latest_total_bytes.total_bytes > 0.5 -- Java is the prominent language of the repo.    
  ) AS pl ON pl.project_id = p.id  
  LEFT JOIN `ghtorrent-bq.ght_2018_04_01.pull_requests` AS pr ON pr.base_repo_id = p.id
  LEFT JOIN `ghtorrent-bq.ght_2018_04_01.pull_request_comments` AS prc ON prc.pull_request_id = pr.id
WHERE p.deleted = FALSE
AND (LOWER(p.LANGUAGE) = 'java' OR LOWER(pl.LANGUAGE) = 'java') -- Java is the prominent language of the repo.
-- To consider whether the repo cannot be a fork?
AND (pc.medium_term_commit_count - pc.recent_commit_count) >= 5
AND (
  (pc.medium_term_distinct_author_count - pc.recent_distinct_author_count) >= 3 
  OR (pc.medium_term_distinct_committer_count - pc.recent_distinct_committer_count) >= 3
) -- At list 3 people working on the repository, involving some degree of collaboration.
AND pc.recent_commit_count >=5 
AND (pc.recent_distinct_author_count >= 3 OR pc.recent_distinct_committer_count >= 3)
AND (prstats.medium_term_pull_request_count - prstats.recent_pull_request_count) >= 5
AND prstats.recent_pull_request_count >= 5
AND LOWER(p.description) NOT LIKE '%mirror of %'
AND prc.comment_id IS NOT NULL -- Need to have records with comments for the analysis.
ORDER BY pc.latest_commit_date DESC

-- Medium term count is overlapping with Recent count.
-- TODO check if the pull requests are actually merged, rather than abandon / closed? Or check the commits if they contains the keyword "Fixes" which actually does merge the pull request? Ans: Doen't matter, as the focus is on the comprehension, not how effective the pull request is.